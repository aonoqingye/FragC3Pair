import os
import re
import json
import base64
import requests
from typing import Dict, List



# ==== LLM PROMPT ==== 组装：把 sample_dict + top_pairs 转成大模型可直接使用的 prompt 文本
def _extract_json_dict(text: str):
    """
    尽力把模型输出里的 JSON 抠出来并解析成 dict：
    1) 优先抓取 ```json ... ``` 代码块
    2) 退化为从第一个 '{' 到最后一个 '}' 的子串
    3) 如仍失败，尝试处理“双重编码”的字符串（再 loads 一次）
    失败则抛出 ValueError
    """
    # 1) ```json ... ``` 代码块
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        cand = m.group(1)
        return json.loads(cand)

    # 2) 第一个左大括号到最后一个右大括号
    l, r = text.find("{"), text.rfind("}")
    if l != -1 and r != -1 and r > l:
        cand = text[l:r+1]
        try:
            return json.loads(cand)
        except Exception:
            # 3) 可能是双重编码: 外层是 JSON 字符串，里层才是目标 JSON
            inner = json.loads(cand)   # cand 本身是个字符串
            if isinstance(inner, str):
                return json.loads(inner)
            return inner

    # 4) 直接就是 JSON
    try:
        return json.loads(text)
    except Exception as e:
        raise ValueError(f"Cannot parse JSON from model output: {e}")


def _encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _merge_dir_to_pair_items(view_vdict: Dict) -> List[Dict]:
    """
    把 shape 形如 [{"dir":"A->B","fragA_id":i,"fragB_id":j,"attn":x, ...}, ...]
    合并成按 (i,j) 聚合的一项，带上 attn_A_to_B / attn_B_to_A 两个方向值。
    """
    merged = {}
    for it in view_vdict.get("top_pairs", []):
        key = (int(it["fragA_id"]), int(it["fragB_id"]))
        obj = merged.setdefault(key, {
            "fragA": it.get("fragA", {"name":"", "smiles":""}),
            "fragB": it.get("fragB", {"name":"", "smiles":""}),
            "attn_A_to_B": "na",
            "attn_B_to_A": "na",
        })
        if it.get("dir") == "A->B":
            obj["attn_A_to_B"] = float(it.get("attn", 0.0))
        elif it.get("dir") == "B->A":
            obj["attn_B_to_A"] = float(it.get("attn", 0.0))
    # 输出最多2条（已在生成端限制），这里保持顺序即可
    out = []
    for (i, j), obj in merged.items():
        out.append({
            "pair_id": f"{i}-{j}",
            "fragA": obj["fragA"],
            "fragB": obj["fragB"],
            "attn_A_to_B": obj["attn_A_to_B"],
            "attn_B_to_A": obj["attn_B_to_A"],
        })
    return out

def _as_name_or_smiles(d: Dict) -> str:
    name = (d or {}).get("name", "").strip()
    smi  = (d or {}).get("smiles", "").strip()
    if name:
        return f"{name} (SMILES: {smi})" if smi else name
    return smi or "unknown"

def _call_llm(
    sample_dict: Dict,
    fig_out_path: str,
    model: str,
    *,
    thresholds: Dict = None,
    external_knowledge: str = "disallowed"
):
    """
    依据提供的模板渲染 Prompt 文本：
    - 默认 external_knowledge = disallowed
    - 缺失名称时要求 LLM 依据 SMILES 推断名称/关键性质
    - thresholds: 例如 {"style":"two_sided","synergistic_if":-10.0,"antagonistic_if":10.0}
                  或   {"style":"one_sided","synergistic_if":-5.0}
    """

    cell_line = sample_dict.get("cell_line", "") or "unknown"
    drugA = sample_dict.get("drugA", {})
    drugB = sample_dict.get("drugB", {})

    # 各视角合并方向
    views = sample_dict.get("views", {})
    fg_items     = _merge_dir_to_pair_items(views.get("fg", {}))
    murcko_items = _merge_dir_to_pair_items(views.get("murcko", {}))
    brics_items  = _merge_dir_to_pair_items(views.get("brics", {}))

    def _pairs_block(items, key):
        lines = []
        for k, it in enumerate(items[:2], 1):
            a_str = _as_name_or_smiles(it.get("fragA"))
            b_str = _as_name_or_smiles(it.get("fragB"))
            a2b   = it.get("attn_A_to_B", "na")
            b2a   = it.get("attn_B_to_A", "na")
            lines.append(
                f'      Fragment_1: "{a_str}"\n'
                f'      Fragment_2: "{b_str}"\n'
                f'      attn_1_to_2: {{value: {a2b}}}\n'
                f'      attn_2_to_1: {{value: {b2a}}}\n'
            )
        if not lines:
            lines.append(f"    # insufficient evidence for {key}\n")
        return "".join(lines)

    # System & Policy
    sys_block = (
        "You are an expert biomedical AI assistant specializing in oncology drug-combination synergy.\n"
        "You must (1) assess the model’s synergy score and (2) explain plausible mechanisms grounded in fragment–fragment cross-attention.\n"
        "Base your explanation on BOTH the structured data in the prompt "
        "AND the attached cross-attention figure. If the text and image disagree, "
        "explain the discrepancy and prioritize the figure for visual evidence.\n"
        'Use only the facts in INPUT unless "external_knowledge" is "allowed". Be cautious, concise, and avoid speculation.\n\n'
        "Policy:\n"
        "- Do not reveal internal reasoning chains. Output only conclusions and concise evidence.\n"
        "- If information is insufficient, say \"insufficient evidence\" precisely where it applies.\n"
        "- Prefer chemistry-grounded terms (functional groups, scaffolds, BRICS blocks), then link to pathway/target-level biology.\n"
        "- When multiple fragment views agree, treat that as stronger evidence.\n"
        "- Infer a reasonable name or key physicochemical property from its SMILES before explanation.\n\n"
    )

    # INPUT
    y_true = sample_dict.get("y_true", None)
    y_pred = sample_dict.get("y_pred", None)
    input_block = (
        "INPUT:\n"
        'task_context: "drug-combination synergy explanation with fragment-level cross-attention"\n'
        f'external_knowledge: "{external_knowledge}"\n'
        "cell_line:\n"
        f'  name: "{cell_line}"\n'
        '  cancer_type: "unknown"\n'
        '  notes: "unknown"\n\n'
        "drugs:\n"
        "  A (Left):\n"
        f'    name: "{drugA.get("id","unknown")}"\n'
        f'    smiles: "{drugA.get("smiles","")}"\n'
        '    targets: "unknown"\n'
        "  B (Right):\n"
        f'    name: "{drugB.get("id","unknown")}"\n'
        f'    smiles: "{drugB.get("smiles","")}"\n'
        '    targets: "unknown"\n\n'
        "synergy_score:\n"
        f"  true value: {y_true}\n"
        f"  predicted value: {y_pred}\n"
        '  comment: "score is the model output for this (DrugA, DrugB, Cell) triplet"\n\n'
        "# Top fragment–fragment pairs per view (max 2 per view). Use names or SMILES; both directions if available.\n"
        "fragment_pairs:\n"
        "  functional_groups:\n"
        f"{_pairs_block(fg_items,'functional_groups')}"
        "  murcko_scaffolds:\n"
        f"{_pairs_block(murcko_items,'murcko_scaffolds')}"
        "  BRICS:\n"
        f"{_pairs_block(brics_items,'BRICS')}\n"
        "attention_notes:\n"
        '  normalization: "per-head reduced to mean (unless configured otherwise); padded tokens removed"\n'
        '  cross_view_agreement_hint: "mention if a similar motif appears across FG/Murcko/BRICS"\n\n'
        "Output format:\n"
        "{\n"
        '  "prediction": "{synergistic | antagonistic | inconclusive}",\n'
        '  "confidence": {0.0_to_1.0},\n'
        '  "decisive_fragment_pairs": [\n'
        '    {\n'
        '      "view": "{functional_groups | murcko_scaffolds | BRICS}",\n'
        '      "pair name": "{fragment_1}, {fragment_2}",\n'
        '      "directionality": "{A (name)->B (name) | B (name)->A (name) | bidirectional}",\n'
        '      "why_decisive": "1-2 sentences: high cross-attention + chemistry rationale + aligns with cell-line context.",\n'
        '      "mechanistic_hypothesis": "Concise mechanism linking fragments to targets/pathways/physicochemical effects."\n'
        "    }\n"
        "  ],\n"
        '  "mechanistic_reasoning": [\n'
        '    "2-4 bullets linking fragments to plausible biophysical/biochemical effects (e.g., H-bond donor/acceptor, cation-π, hydrophobic core, kinase hinge binding, DHFR inhibition, MEK pathway blockade).",\n'
        '    "Explain complementarity between Drug A and Drug B in the given cell line (e.g., pathway co-inhibition, synthetic lethality, overcoming resistance)."\n'
        "  ],\n"
        '  "cross_view_consistency": "Summarize whether FG/Murcko/BRICS highlight the same motif(s) or complementary ones.",\n'
        '  "cell_line_context": "State how known features (if any) make the hypothesis more/less plausible; say \'insufficient evidence\' if unknown.",\n'
        '  "risks_or_counterevidence": "Note any antagonism risks (e.g., PK/PD conflicts) or uncertainty.",\n'
        "}\n"
        'Respond with STRICT JSON only.\n'
        'No code fences, no markdown, no natural language, no trailing text.\n'
    )

    img_b64 = _encode_image_b64(fig_out_path)

    # ===================== Responses API Payload =====================
    # 仍然叫 prompt，这样上游 (prompt, parsed) 不用改
    prompt = {
        "model": model,   # 比如 "gpt-5.1" / "gpt-5.1-vision-preview"
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": sys_block,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": input_block,
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{img_b64}",
                    },
                ],
            },
        ],
        "temperature": 0.2,
        "max_output_tokens": 2048,
        # Responses API 中，JSON 输出通过 text.format 控制
        "text": {
            "format": "json"
        },
    }

    url = "https://api.openai.com/v1/chat/completions"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY environment variable")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(url, headers=headers, data=json.dumps(prompt))
    resp.raise_for_status()
    rjson = resp.json()

    # Responses API: output[0].content[0].text
    try:
        output_list = rjson["output"]
        content_list = output_list[0]["content"]
        response_text = content_list[0]["text"]
    except Exception as e:
        raise RuntimeError(f"Unexpected Responses API payload: {rjson}") from e

    # 统一转成 dict
    if isinstance(response_text, str):
        parsed = _extract_json_dict(response_text)
    else:
        parsed = response_text

    return prompt, parsed

