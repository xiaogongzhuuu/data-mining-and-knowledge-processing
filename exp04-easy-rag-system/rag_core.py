import streamlit as st
import requests
from config import MAX_NEW_TOKENS_GEN, TEMPERATURE, TOP_P, OLLAMA_BASE_URL

def generate_answer(query, context_docs, gen_model, tokenizer):
    """Generates an answer using Ollama API based on query and context."""
    if not context_docs:
        return "抱歉，我在数据库中找不到相关文档来回答您的问题。"
    if not gen_model:
         st.error("Generation model not available.")
         return "Error: Generation components not loaded."

    # 构建上下文，包含文档标题和来源信息
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        title = doc.get('title', '未知标题')
        content = doc.get('content', '')[:1200]  # 增加上下文长度到1200字符
        source = doc.get('source_file', '未知来源')
        context_parts.append(f"""
【文档 {i}】
标题：{title}
来源：{source}
内容：{content}
""")

    context = "\n".join(context_parts)

    # 改进的提示词工程
    prompt = f"""你是一位专业的中医医疗问答助手，擅长根据医疗文献回答用户问题。

【任务要求】
1. 仔细阅读以下上下文文档，理解其中的医疗知识和观点
2. 根据文档内容准确回答用户的问题
3. 如果文档中有明确的答案，请直接引用或总结
4. 如果文档中没有相关信息，请明确说明"提供的文档中没有相关信息"
5. 不要编造文档中没有的内容
6. 使用专业、准确的医学术语
7. 回答要条理清晰，结构完整

【上下文文档】
{context}

【用户问题】
{query}

【回答格式】
请按照以下格式回答：

**回答：**
[你的答案内容]

**参考来源：**
- [文档1标题]
- [文档2标题]
（仅列出实际引用的文档）

现在请回答："""

    try:
        # 调用 ollama API
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": gen_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": MAX_NEW_TOKENS_GEN,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P
                }
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "").strip()
            
            # 如果答案为空，返回默认提示
            if not answer:
                return "抱歉，系统无法生成答案，请稍后重试。"
            
            return answer
        else:
            st.error(f"Ollama API error: {response.status_code}")
            return f"抱歉，生成答案时遇到错误（HTTP {response.status_code}），请稍后重试。"
    except requests.exceptions.Timeout:
        st.error("请求超时")
        return "抱歉，生成答案超时，请稍后重试。"
    except requests.exceptions.ConnectionError:
        st.error("连接Ollama服务失败")
        return "抱歉，无法连接到生成服务，请检查Ollama服务是否正常运行。"
    except Exception as e:
        st.error(f"Error during text generation: {e}")
        return f"抱歉，生成答案时遇到错误：{str(e)}" 