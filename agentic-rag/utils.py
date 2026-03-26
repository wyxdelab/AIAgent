from pathlib import Path
import config
import pymupdf4llm
import pymupdf
import re
import tiktoken

def pdf_to_markdown(path, overwrite = False):
    output_dir = Path(config.MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = (Path(output_dir) / Path(path).stem).with_suffix('.md')
    if overwrite or not md_path.exists():
        doc = pymupdf.open(path)
        md = pymupdf4llm.to_markdown(doc, header=False, footer=False, page_separators=False, ignore_images=True, write_images=False, image_path=None)
        md_cleaned = clean_pymupdf4llm_noise(md).encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
        output_path = (Path(output_dir) / Path(doc.name).stem).with_suffix('.md')
        Path(output_path).write_bytes(md_cleaned.encode('utf-8'))

def clean_pymupdf4llm_noise(md: str) -> str:
    # 1) 删掉图片占位行：**==> picture [w x h] intentionally omitted <==**
    md = re.sub(
        r"(?im)^\s*\*\*==>\s*picture\s*\[\d+\s*x\s*\d+\]\s*intentionally omitted\s*<==\*\*\s*\n?",
        "",
        md
    )

    # 2) 删掉整段图片文本块（含中间内容）
    md = re.sub(
        r"(?is)\*\*-----\s*Start of picture text\s*-----\*\*<br>\s*.*?\s*\*\*-----\s*End of picture text\s*-----\*\*<br>\s*\n?",
        "",
        md
    )

    # 3) 压缩多余空行（可选）
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    return md

def estimate_context_tokens(message: list) -> int:
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens_len = []
    for msg in message:
        if hasattr(msg, "content") and msg.content:
            tokens_len.append(len(encoding.encode(str(msg.content))))
    return sum(tokens_len)
