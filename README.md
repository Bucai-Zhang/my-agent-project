# Lightweight Tool-Calling RAG Agent

一个用于学习和实践 **LLM Agent、Tool Calling、Memory 与 RAG（Retrieval-Augmented Generation）** 的轻量级项目。

> 当前项目定位：**教学型原型 / 学习型工程项目**。  
> 已经跑通 Agent 与 RAG 的核心链路，但仍在继续工程化，包括多文档知识库、向量索引、RAG 与 Agent 统一、评测与日志等。

---

## 1. 项目目标

这个项目的目标不是调用一次大模型 API 得到答案，而是逐步理解并实现一个 Agent 系统的核心组成部分：

- 大语言模型如何调用外部工具
- Agent Loop 如何工作
- 如何保存和压缩对话历史
- 如何读取、搜索本地文件
- 如何把长文档切分成多个 Chunk
- 如何通过 Embedding 实现语义检索
- 如何通过 Top-K 与相似度阈值筛选相关资料
- 如何将检索结果作为上下文交给 LLM 生成最终回答

目前主要使用：

- Python
- DeepSeek API
- OpenAI Python SDK（DeepSeek API 兼容接口）
- Sentence Transformers
- `BAAI/bge-small-zh-v1.5`
- Git

---

## 2. 当前已经实现的功能

### 2.1 Tool Calling

Agent 当前支持多个工具，例如：

- `calculator`：加减乘除
- `text_length`：计算文本长度
- `read_file`：读取文本文件
- `list_files`：列出目录内容
- `search_file`：递归查找文件

LLM 不直接执行 Python 函数，而是先返回一个 `function_call`，Python 根据工具名称和参数执行真实函数，再通过 `function_call_output` 把结果返回给 LLM。

基本流程：

```text
User
  ↓
LLM 判断是否需要工具
  ↓
function_call
  ↓
Python 执行工具
  ↓
function_call_output
  ↓
LLM 继续推理
  ↓
Final Answer
```

---

### 2.2 Agent Loop

项目实现了一个简单的 Agent Loop：

```text
调用 LLM
  ↓
是否产生 tool call？
  ├─ 是 → 执行工具 → 把结果放回上下文 → 再次调用 LLM
  └─ 否 → 返回最终答案
```

因此一个任务可以连续调用多个工具。

例如：

```text
用户：帮我找到 test.txt 并读取内容

search_file
  ↓
得到文件路径
  ↓
read_file
  ↓
读取文件
  ↓
LLM 生成最终回答
```

---

### 2.3 对话记忆

项目目前有两层简单记忆机制：

- `conversation_history`：保存最近的用户与助手消息
- `memory_summary`：将较早的聊天记录压缩成摘要

这样可以避免把全部历史对话一直塞给模型。

当前思路：

```text
较早历史 → 摘要 memory_summary
最近若干轮 → 保留原始消息
当前问题 → 加入工作上下文
```

---

## 3. RAG 实现

RAG 全称：

**Retrieval-Augmented Generation（检索增强生成）**

核心思想不是重新训练大模型，而是：

```text
先查资料
  ↓
找到和问题相关的内容
  ↓
把内容交给 LLM
  ↓
让 LLM 基于资料回答
```

---

### 3.1 文档切块 Chunking

长文档不能简单一次全部放入上下文，因此先切成多个文本块：

```python
def split_text(text, chunk_size=1000, overlap=100):
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)

    return chunks
```

其中：

- `chunk_size`：每个文本块最大长度
- `overlap`：相邻文本块重复的内容长度

使用 overlap 是为了降低关键信息刚好被切在两个 Chunk 边界上的风险。

---

### 3.2 Embedding

关键词搜索存在明显缺陷。

例如：

```text
文档：Git 是一种版本控制工具
问题：怎么管理代码版本？
```

虽然两句话语义相关，但字面并不相同。

因此项目使用 Embedding 将文本转换为向量：

```text
文本
  ↓
Embedding Model
  ↓
512 维向量
```

当前使用：

```text
BAAI/bge-small-zh-v1.5
```

---

### 3.3 余弦相似度

项目手动实现了余弦相似度，用于比较 Query 与 Chunk 的向量：

```python
def cosine_similarity(a, b):
    dot_product = 0
    a_length = 0
    b_length = 0

    for x, y in zip(a, b):
        dot_product += x * y
        a_length += x * x
        b_length += y * y

    return dot_product / ((a_length ** 0.5) * (b_length ** 0.5))
```

基本思想：

```text
语义越接近
→ Embedding 向量方向通常越接近
→ Cosine Similarity 越高
```

---

### 3.4 Top-K Retrieval

不是只返回相似度最高的一个 Chunk，而是：

```text
Query
  ↓
与所有 Chunk 比较
  ↓
按照相似度排序
  ↓
返回 Top-K
```

当前实现会把每个结果保存为：

```python
(chunk, score)
```

然后按 `score` 从高到低排序。

---

### 3.5 Threshold

Top-K 只能保证：

> 这是所有文本里最相似的几个。

并不能保证：

> 它们真的和问题相关。

因此项目增加了相似度阈值：

```python
threshold = 0.4
```

只有相似度超过阈值的文本块才进入最终 Context。

如果没有任何 Chunk 通过阈值，则返回：

```text
参考资料中没有找到足够相关的信息。
```

> 当前 `0.4` 只是实验阶段的经验参数，并不是固定最优值，后续需要通过评测数据调整。

---

### 3.6 Generation

筛选得到的 Chunk 会被拼接成 Context：

```text
Chunk 1

Chunk 2

Chunk 3
```

然后连同用户问题一起交给 DeepSeek：

```text
System:
请根据提供的参考资料回答问题。
如果资料中没有答案，就明确说资料中没有相关信息。

User:
参考资料：
<retrieved context>

问题：
<query>
```

最终流程：

```text
Document
  ↓
Chunking
  ↓
Embedding
  ↓
Vector Retrieval
  ↓
Top-K
  ↓
Threshold
  ↓
Context
  ↓
LLM
  ↓
Answer
```

---

## 4. 为什么提前计算 Chunk Embedding？

最早的实现中，每次用户提问都会重新计算所有 Chunk 的 Embedding：

```text
问题1 → 所有 Chunk 重新 Embedding
问题2 → 所有 Chunk 重新 Embedding
问题3 → 所有 Chunk 重新 Embedding
```

但文档没有变化时，这些计算是重复的。

因此现在改为：

```text
程序启动
  ↓
Chunk Embedding 只计算一次
  ↓
保存 chunk_vectors

用户问题
  ↓
只计算 query_vector
  ↓
和已有 chunk_vectors 比较
```

这样减少重复计算，提高检索效率。

---

## 5. 项目结构

当前主要结构：

```text
agent_project/
│
├── main.py
│   └── 用户交互入口
│
├── agent.py
│   ├── Agent Loop
│   ├── Tool Calling
│   ├── Memory
│   └── RAG Answer Generation
│
├── rag.py
│   ├── 加载 Embedding Model
│   ├── 文档预处理
│   ├── 文档向量
│   └── RAG Query
│
├── tools.py
│   ├── calculator
│   ├── text_length
│   ├── read_file
│   ├── list_files
│   ├── search_file
│   ├── split_text
│   ├── cosine_similarity
│   └── semantic_search
│
├── tool_schema.py
│   └── Tool Schema
│
├── config.py
│   └── API Configuration
│
└── text.txt
    └── 当前测试知识库
```

---

## 6. 当前项目的局限

目前项目仍然是一个学习型原型，存在以下明显不足：

1. 当前知识库主要还是单个文本文件。
2. 文本切块还是简单字符切分，没有使用更复杂的语义切块。
3. Chunk Vector 只保存在内存，没有持久化。
4. 没有接入 FAISS、Milvus、Chroma、pgvector 等向量检索系统。
5. Threshold 仍然是手动设置，还没有经过系统评测。
6. RAG 与原来的 Tool-Calling Agent 还没有完全统一。
7. 没有正式的 Retrieval / Answer Evaluation。
8. 暂未支持 PDF、Markdown、网页等多种文档。
9. 暂未实现 Citation / Source Trace。
10. 目前主要关注理解原理，还没有进行性能优化。

---

## 7. 后续计划

接下来计划逐步完成：

- [ ] 支持指定任意知识库文件
- [ ] 多文档知识库
- [ ] 建立 Indexing 流程
- [ ] Chunk Vector 持久化
- [ ] 引入向量索引
- [ ] RAG 作为 Agent 的一个 Tool
- [ ] Agent 自动决定是否调用 RAG
- [ ] 支持 PDF / Markdown / 网页
- [ ] 增加 Citation
- [ ] 增加日志与错误处理
- [ ] 建立 RAG 测试集
- [ ] 比较 Chunk Size / Overlap / Top-K / Threshold
- [ ] 比较不同 Embedding 模型
- [ ] 增加 RAG 效果与性能评测

---

# 8. 面试 / 汇报可能被问到的问题

下面的问题不是为了死记答案，而是用于复习项目原理。面试时应尽量结合实际代码和实验回答。

---

## Q1：你这个项目到底是什么？

**参考回答：**

这是一个我从底层逐步实现的轻量级 LLM Agent 与 RAG 原型。

Agent 部分实现了 Tool Calling、Agent Loop、简单的对话记忆以及文件操作工具；RAG 部分实现了文档切块、Embedding、余弦相似度、Top-K 检索、相似度阈值过滤以及最终基于检索 Context 的生成。

当前主要目标是理解 Agent 和 RAG 的核心工作机制，而不是直接依赖完整框架把所有功能封装掉。后续会继续做多文档、向量索引、评测，以及把 RAG 作为 Agent 的一个工具统一进去。

---

## Q2：为什么你没有直接用 LangChain？

**参考回答：**

因为这个阶段我的主要目标是理解底层流程。

例如 Tool Calling 中，我希望真正理解：

```text
模型产生 function_call
→ Python 执行工具
→ function_call_output 返回模型
→ 模型继续推理
```

RAG 中我也希望自己实现：

```text
Chunk
→ Embedding
→ Similarity
→ Top-K
→ Context
→ Generation
```

如果一开始直接使用高度封装的框架，虽然开发速度快，但容易只会调用接口，不理解内部发生了什么。

后续在底层流程理解清楚以后，我会再学习成熟框架，并比较它们和当前实现的差异。

---

## Q3：Tool Calling 是模型直接调用 Python 函数吗？

**参考回答：**

不是。

LLM 本身不会直接运行本地 Python 函数。

模型会生成一个结构化的 Tool Call，例如：

```text
name = calculator
arguments = {"a": 1, "b": 2, "op": "+"}
```

Python 程序解析参数以后，通过 `tool_map` 找到真正的函数并执行：

```python
tool = tool_map.get(item.name)
result = tool(**arguments)
```

然后把执行结果作为 `function_call_output` 重新交给模型。

所以真实工具执行发生在 Python 程序里，而不是 LLM 内部。

---

## Q4：为什么需要 Agent Loop？

**参考回答：**

因为一个任务不一定只需要一次工具调用。

例如：

```text
找到文件
→ 读取文件
→ 根据内容继续处理
```

一次 LLM 调用可能只决定下一步要做什么。

因此需要循环：

```text
LLM
→ Tool Call
→ Tool Result
→ LLM
→ ...
→ Final Answer
```

直到模型不再产生工具调用，再返回最终答案。

---

## Q5：你的 Memory 是怎么实现的？

**参考回答：**

目前使用了一个比较简单的两层方案：

- 最近若干轮对话保留原始消息；
- 较早的消息使用 LLM 压缩成 `memory_summary`。

因此实际上下文大致是：

```text
System Prompt
+ 历史摘要
+ 最近对话
+ 当前问题
```

这样做的目的主要是减少上下文长度，同时尽量保留重要历史信息。

目前这个 Memory 还是内存级的，程序重启以后不会持久保存，后面可以进一步做持久化。

---

## Q6：什么是 RAG？

**参考回答：**

RAG 是 Retrieval-Augmented Generation，也就是检索增强生成。

它的核心不是重新训练 LLM，而是在回答问题之前：

1. 从外部知识库检索和问题相关的内容；
2. 把这些内容作为 Context；
3. 再让 LLM 基于 Context 生成回答。

因此可以让模型使用训练数据之外的私有或更新知识，同时降低直接依靠模型参数回答造成的幻觉风险。

---

## Q7：为什么不能直接把整个文档交给 LLM？

**参考回答：**

主要有几个原因：

第一，文档可能超过模型上下文窗口。

第二，即使没有超过，大量无关内容也会增加 Token 成本和推理负担。

第三，过长 Context 可能使真正重要的信息被大量无关内容淹没。

所以一般先切块，再检索真正相关的部分。

---

## Q8：为什么需要 Chunk？

**参考回答：**

因为检索通常不是以整篇长文档为最小单位，而是需要找到其中局部最相关的信息。

如果整篇文档只对应一个 Embedding，那么不同主题会被压缩在同一个向量里，检索粒度太粗。

所以先把长文档拆成多个 Chunk，再分别建立 Embedding。

---

## Q9：为什么需要 overlap？

**参考回答：**

因为简单的固定长度切块可能把一个完整语义单元切在两个 Chunk 之间。

比如一个重要句子前半部分在 Chunk A，后半部分在 Chunk B。

加入 overlap 后，相邻 Chunk 会重复一部分内容，从而降低边界切断重要信息造成的检索损失。

---

## Q10：Embedding 是什么？

**参考回答：**

Embedding 可以把文本映射成一个高维向量。

在当前项目中，`BAAI/bge-small-zh-v1.5` 会把文本表示成 512 维向量。

单独看每一维通常没有直观的人类语义，但整个向量共同表示文本的语义特征。

语义相近的文本通常在向量空间中也更接近，因此可以用于语义检索。

---

## Q11：为什么不用关键词检索？

**参考回答：**

关键词检索只能很好地处理字面匹配。

例如：

```text
文档：Git 是一种版本控制工具
问题：怎么管理代码版本？
```

虽然语义相关，但关键词可能不同。

Embedding 检索比较的是语义向量，因此即使表达方式不同，也可以找到相关内容。

关键词检索并不是没有价值，实际系统中也可以把关键词检索和向量检索组合成 Hybrid Search。

---

## Q12：Cosine Similarity 是什么？

**参考回答：**

余弦相似度通过比较两个向量之间的夹角来衡量方向是否接近。

公式本质是：

```text
点积
--------------------
向量A长度 × 向量B长度
```

相比直接比较欧氏距离，它更关注向量方向。

文本 Embedding 经常通过余弦相似度衡量语义接近程度。

---

## Q13：为什么要 Top-K？

**参考回答：**

如果只取最相似的一个 Chunk，可能丢失跨 Chunk 信息。

因此通常返回最相关的前 K 个文本块，再一起提供给 LLM。

但是 K 也不是越大越好。

K 太大可能：

- 增加 Token
- 引入不相关信息
- 干扰模型回答

所以 Top-K 是需要实验调整的超参数。

---

## Q14：既然已经 Top-K，为什么还要 Threshold？

**参考回答：**

Top-K 只能保证返回“所有候选中相对最相似的 K 个”，不能保证它们真的相关。

例如文档完全没有用户需要的信息，Top-1 仍然存在。

因此增加 Threshold：

```text
Similarity < Threshold
→ 不认为是有效知识
```

如果所有候选都低于阈值，则直接告诉用户知识库里没有足够相关的信息，而不是强行把无关内容交给 LLM。

---

## Q15：为什么 Threshold 是 0.4？0.4 是固定的吗？

**参考回答：**

不是固定的。

目前 `0.4` 只是开发测试阶段的经验参数。

真正的阈值应该通过验证集或测试问题进行评测，例如观察：

- Relevant Retrieval Recall
- Precision
- Answer Accuracy
- False Rejection
- False Retrieval

然后选择适合当前 Embedding 模型、知识库和任务的数据驱动阈值。

---

## Q16：为什么要提前计算 Chunk Embedding？

**参考回答：**

因为知识库不变时，Chunk Embedding 也是不变的。

如果每个 Query 都重新计算所有 Chunk 的 Embedding，会产生大量重复计算。

所以当前实现是在程序初始化时：

```text
Document
→ Chunk
→ Chunk Embedding
```

只做一次。

查询阶段只需要：

```text
Query
→ Query Embedding
→ Similarity Search
```

后续还可以进一步将向量持久化到 Vector Index / Vector Database 中。

---

## Q17：现在的向量检索有什么性能问题？

**参考回答：**

当前实现是暴力遍历：

```text
Query Vector
→ 和每一个 Chunk Vector 计算相似度
```

时间复杂度大致随着 Chunk 数量线性增长。

文档少的时候没有问题，但知识库很大时效率不够。

后续可以使用 FAISS、HNSW 或向量数据库进行近似最近邻搜索，提高大规模检索效率。

---

## Q18：为什么你使用 bge-small-zh-v1.5？

**参考回答：**

目前主要考虑三个因素：

1. 对中文语义检索比较友好；
2. 模型规模相对较小，本地运行成本低；
3. 适合当前学习型 RAG 原型快速实验。

当前选择并不代表它一定是最终最优模型。

后续可以比较不同 Embedding 模型的检索质量、推理速度和资源消耗。

---

## Q19：DeepSeek 在这个项目里负责什么？

**参考回答：**

当前 DeepSeek 主要承担 LLM 推理和最终生成。

例如：

- Tool Calling Agent 中负责判断是否调用工具、生成工具参数和最终回答；
- RAG 中负责根据检索到的 Context 和 Query 生成最终答案；
- Memory 中也用于把较早的历史对话压缩成摘要。

Embedding 当前不是通过 DeepSeek，而是使用本地 `bge-small-zh-v1.5`。

---

## Q20：为什么使用 OpenAI Python SDK，但模型是 DeepSeek？

**参考回答：**

因为 DeepSeek API 支持 OpenAI 兼容接口。

因此可以使用：

```python
from openai import OpenAI
```

创建客户端，然后通过 DeepSeek 的 `base_url` 和 API Key 访问 DeepSeek 模型。

SDK 只是客户端工具，不代表实际模型一定来自 OpenAI。

---

## Q21：System / User / Assistant Role 有什么区别？

**参考回答：**

我目前这样理解：

- `system`：定义模型的工作规则、行为边界和角色；
- `user`：当前用户的输入或任务；
- `assistant`：模型已经产生的历史回复。

例如 RAG 中：

```text
System：
请根据参考资料回答，没有答案就明确说明。

User：
参考资料：...
问题：...
```

这样把“规则”和“当前任务”分开。

---

## Q22：你这个项目最大的不足是什么？

**参考回答：**

目前最大的不足是它仍然属于原理验证型 RAG / Agent，而不是成熟系统。

例如：

- Knowledge Base 规模很小；
- 向量仍然主要保存在内存；
- Retrieval 算法是线性扫描；
- 没有系统评测；
- Agent 和 RAG 尚未完全统一；
- 文档解析能力比较有限；
- 还没有 Citation 和完整的 Observability。

所以后续工作重点不是继续堆 Demo 功能，而是进行工程化和评测。

---

## Q23：如果让你下一步继续做，你优先做什么？

**参考回答：**

我会优先做三个方向：

第一，把知识库构建和 Query 解耦，支持多文档 Indexing 与持久化。

第二，把 RAG 作为 Agent 的一个 Tool，让 Agent 根据任务自主决定是否检索知识库，而不是所有问题都强制进入 RAG。

第三，建立一套小型评测集，对 Chunk Size、Overlap、Top-K、Threshold 和不同 Embedding Model 进行实验，而不是只凭感觉调参数。

---

## Q24：这个项目哪些是你自己实现的？

**参考回答：**

目前我自己实现和理解的主要部分包括：

- Agent Loop
- Tool Map 和工具执行流程
- Tool Calling 返回结果的处理
- 简单 Conversation Memory 与 Rolling Summary
- 文件读取和搜索工具
- 固定长度 Chunking 与 Overlap
- 关键词检索
- Cosine Similarity
- 基础 Semantic Retrieval
- Top-K 排序
- Threshold Filtering
- Retrieval Context 拼接
- RAG Generation Pipeline

Embedding 模型和 LLM 本身使用已有开源模型 / API，我没有声称这些模型是自己训练的。

---

## Q25：这个项目和直接调用 ChatGPT / DeepSeek 有什么区别？

**参考回答：**

直接调用 LLM 时，模型主要依赖自身参数和当前 Prompt。

这个项目在 LLM 外面增加了一层程序系统，包括：

```text
Memory
Tools
Retrieval
External Data
Control Loop
```

所以模型不再只是“一问一答”，而是可以：

- 根据任务调用工具；
- 获取外部信息；
- 使用私有知识库；
- 结合历史上下文；
- 多步完成任务。

---

# 9. 建议复习重点

如果准备面试，优先确保自己可以不看答案解释清楚下面这些：

```text
1. Tool Calling 到底是谁执行工具？
2. 为什么需要 Agent Loop？
3. Memory Summary 为什么存在？
4. RAG 为什么不是训练模型？
5. Chunk / Overlap 分别解决什么问题？
6. Embedding 是什么？
7. 为什么 Cosine Similarity 可以做语义检索？
8. Top-K 和 Threshold 分别解决什么问题？
9. 为什么 Chunk Embedding 要提前计算？
10. 当前暴力向量检索的问题是什么？
11. 你的项目目前有哪些真实不足？
12. 下一步怎么把 Demo 做成更完整项目？
```

如果这些问题能用自己的话讲清楚，说明你已经不只是“代码跑通了”，而是真的理解了当前阶段项目。

---

## 10. 声明

本项目目前主要用于学习 Agent / RAG 的底层工作机制。

项目中使用的 DeepSeek API、OpenAI Python SDK、Sentence Transformers 和 BGE Embedding Model 均为已有工具或模型。项目重点在于自己实现和理解 Agent 控制流程、Tool Calling、Memory、Retrieval 与 RAG Pipeline，并在此基础上继续进行工程化、实验和改进。
