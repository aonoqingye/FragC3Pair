# ======== LLM Explainer (text + image) ========
import os, json, base64, argparse, glob
from typing import Optional, Dict, Any

def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_gpt_with_image(prompt_text: str, image_path: str,
                        model: str = "gpt-4", #"gpt-4" version in gpt-4o-2024-08-06, "gpt-4-turbo" version in gpt-4-vision
                        temperature: float = 0.2,
                        max_tokens: int = 2048) -> Dict[str, Any]:
    """
    学校示例的“文本 + 图片”请求体。返回：尽力解析成 JSON 的 dict。
    若解析失败，返回 {"raw": <原始模型输出>} 以便排查。
    """
    # 延迟导入，避免对项目主流程造成依赖干扰

    img_b64 = _encode_image_b64(image_path)
    user_content = [
        {"type": "text", "text": prompt_text},
        {
            "type": "image_url",
            "image_url": {
                # 使用 data URL，避免外网地址依赖
                "url": f"data:image/png;base64,{img_b64}"
            },
        },
    ]

    if OpenAI is not None:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": user_content}],
        )
        out = resp.choices[0].message.content
    else:
        # 旧版 openai.ChatCompletion.create
        out = openai.ChatCompletion.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": user_content}],
        )["choices"][0]["message"]["content"]

    # 容错 JSON 解析
    try:
        first_brace = out.find("{")
        last_brace  = out.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return json.loads(out[first_brace:last_brace+1])
        return {"raw": out}
    except Exception:
        return {"raw": out}

def explain_one(prompt_path: str, image_path: str,
                model: str = "gpt-4o-mini",
                out_json: Optional[str] = None) -> str:
    """
    单样本解释：读取 prompt 与图片 -> 调 GPT -> 写 JSON 结果文件。
    返回结果文件路径。
    """
    prompt_text = _read_text(prompt_path)

    result = call_gpt_with_image(prompt_text, image_path, model=model)
    if out_json is None:
        # 与单样本图同名，后缀 .llm.json
        stem = os.path.splitext(image_path)[0]
        out_json = f"{stem}.llm.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return out_json

def _pair_prompt_image_under_dir(work_dir: str):
    """
    在目录下将 *.prompt.txt 与 *.png 做 1:1 配对（同名前缀）。
    返回 [(prompt_path, image_path)]。
    """
    prompts = sorted(glob.glob(os.path.join(work_dir, "*.prompt.txt")))
    images  = sorted(glob.glob(os.path.join(work_dir, "*.png")))
    img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in images}
    pairs = []
    for pr in prompts:
        stem = os.path.basename(pr).replace(".prompt.txt", "")
        if stem in img_map:
            pairs.append((pr, img_map[stem]))
    return pairs

def main_llm():
    ap = argparse.ArgumentParser(description="LLM explainer (text+image). Minimal intrusion runner.")
    ap.add_argument("--prompt", type=str, default=None, help="单个 .prompt.txt 文件路径")
    ap.add_argument("--image",  type=str, default=None, help="单个 .png 可视化图路径")
    ap.add_argument("--dir",    type=str, default=None, help="目录：批量配对 *.prompt.txt 与 *.png")
    ap.add_argument("--model",  type=str, default="gpt-4",
                    help="可换 gpt-4 (gpt-4o-2024-08-06), gpt-4-turbo (gpt-4-vision)")
    args = ap.parse_args()

    tasks = []
    if args.prompt and args.image:
        tasks = [(args.prompt, args.image)]
    elif args.dir:
        tasks = _pair_prompt_image_under_dir(args.dir)

    if not tasks:
        raise SystemExit("No prompt/image pair found. Specify --prompt & --image or --dir.")

    for pr, im in tasks:
        outp = explain_one(pr, im, model=args.model)
        print(f"[LLM] {os.path.basename(im)} -> {outp}")

# 允许作为脚本独立运行：python explainer.py --dir attn_pair/fold_0
if __name__ == "__main__" and os.getenv("RUN_EXPLAINER", "0") == "1":
    main_llm()
