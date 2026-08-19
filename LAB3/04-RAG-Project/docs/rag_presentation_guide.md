# คู่มือเตรียมตัวสอบและนำเสนอสไลด์ RAG (DL-04) 🎓
**กำหนดการนำเสนอ:** 19 สิงหาคม 2569 (2026)  
**หัวข้อ:** การพัฒนาระบบ RAG ขั้นสูง (Advanced RAG System) ด้วยดาราศาสตร์และการสำรวจอวกาศ 🌌

---

## 1. ข้อมูลภาพรวมระบบ (RAG System Overview)
ระบบที่พัฒนาขึ้นใน **DL-04** (โฟลเดอร์ `LAB3/04-RAG-Project`) คือ **Advanced RAG System** ที่ใช้สำหรับถาม-ตอบข้อมูลดาราศาสตร์และการสำรวจอวกาศเป็นภาษาไทย
* **จุดเด่นที่เป็น Advanced RAG**: ไม่ได้มีเพียงแค่การค้นหาเวกเตอร์ทั่วไป (Dense Retrieval) แต่มีขั้นตอนเพิ่มประสิทธิภาพครบครัน ได้แก่ **Query Transformation** (ปรับแต่งคำถาม), **Hybrid Search** (คีย์เวิร์ด + เวกเตอร์), **RRF** (จัดอันดับคะแนนผสม), **Reranking** (คัดกรองความหมายอย่างละเอียด) และ **Conversation Memory** (จดจำบทสนทนาโต้ตอบต่อเนื่อง)

---

## 2. โครงสร้างสไลด์นำเสนอ (1–2 สไลด์)

### สไลด์ที่ 1: Architecture & RAG Process Workflow (หัวใจหลัก)
ให้วาดแผนภาพ (Diagram) แสดงโฟลว์จากอินพุตสู่เอาต์พุตตามลำดับขั้นตอน (Process Flow) ดังนี้:

```mermaid
graph TD
    A[1. User Input คำถามผู้ใช้] --> B(2. Conversation Memory ดึงประวัติ)
    B --> C(3. Query Transformer ปรับแต่งคำถาม)
    
    subgraph Retrieval Stage
        C --> D[Dense Search / FAISS]
        C --> E[Sparse Search / BM25]
        D --> F[Reciprocal Rank Fusion RRF]
        E --> F
    end

    F --> G(4. Reranker / Cross-Encoder)
    G --> H(5. Context Builder จัดรูปข้อมูลอ้างอิง)
    H --> I(6. Generator / LLM)
    I --> J[7. User Output คำตอบพร้อมแหล่งอ้างอิง [n]]
```

* **รายละเอียดข้อความที่จะเขียนลงสไลด์ 1**:
  - **Input**: คำถามภาษาไทย เช่น *"ดาวศุกร์ทำไมร้อนที่สุด"*
  - **Query Transformation**: ปรับแต่งคำถาม/คำสะกดผิด หรือแปลงคำถามอ้างอิงจากบริบทประวัติเก่า (ด้วยโหมด Multi-Query หรือ HyDE)
  - **Hybrid Search**: ค้นหาพร้อมกัน 2 แบบเพื่ออุดจุดอ่อนซึ่งกันและกัน (Dense Search ผ่าน FAISS หาสารัตถะความหมาย + Sparse Search ผ่าน BM25 หาคำเฉพาะทางวิชาการและตัวเลข)
  - **Reciprocal Rank Fusion (RRF)**: อัลกอริทึมคณิตศาสตร์รวมผลการค้นหาจากทั้งสองคลังเข้าด้วยกันโดยไม่ใช้คะแนนดิบ แต่ใช้ "อันดับ (Rank)"
  - **Reranker**: โมเดลระดับสูง (Cross-Encoder) มาจัดอันดับความเกี่ยวข้องของเอกสารที่เข้ารอบ 20 ชิ้นแรกใหม่อีกครั้งอย่างแม่นยำ
  - **LLM Context Generation**: ส่งเอกสารอันดับที่ดีที่สุด 3 ชิ้นเข้าสู่ Prompt เพื่อให้ LLM เขียนตอบอย่างน่าเชื่อถือพร้อมระบุแหล่งอ้างอิงรายประโยค เช่น `[1]`, `[2]`

### สไลด์ที่ 2: Source Code Mapping & Evaluation (การเชื่อมโยงโค้ดและผลการวัดผล)
* **ตารางสรุปขั้นตอนด่าน RAG กับไฟล์และฟังก์ชันที่เกี่ยวข้อง**:
  *(ตารางนี้ใช้ตอบคำถามอาจารย์แบบ 1:1 ได้ทันที)*

| ขั้นตอนใน Workflow | ไฟล์ซอร์สโค้ด (Source Code File) | ฟังก์ชันหลัก (Core Function) |
| :--- | :--- | :--- |
| **RAG Pipeline (ภาพรวม)** | [`src/rag_pipeline.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/rag_pipeline.py) | `RAGPipeline.ask()` |
| **1. Query Transformation** | [`src/query_transform.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/query_transform.py) | `QueryTransformer.transform()` |
| **2. Conversation Memory** | [`src/memory.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/memory.py) | `ConversationMemory.get_context()` |
| **3. Hybrid Retrieval** | [`src/hybrid_retriever.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/hybrid_retriever.py) | `HybridRetriever.retrieve()` |
| **4. Reciprocal Rank Fusion** | [`src/hybrid_retriever.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/hybrid_retriever.py) | `reciprocal_rank_fusion()` |
| **5. Reranker** | [`src/rerankers.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/rerankers.py) | `Reranker.rerank()` |
| **6. Prompt & Context Builder**| [`src/prompt_templates.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/prompt_templates.py) | `build_messages()`, `format_context()` |
| **7. LLM Generator** | [`src/generator.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/generator.py) | `Generator.generate()` |

* **ผลการวัดผลระบบ (Evaluation Metrics)**:
  ผลจากการรันสคริปต์ `evaluation.eval_retrieval` บนชุดข้อสอบดาราศาสตร์ (`golden_set.json`):
  - **Hit@1 (อัตราเจอเอกสารเป้าหมายในอันดับที่ 1)**: `1.0000` (100%)
  - **MRR (Mean Reciprocal Rank)**: `1.0000`
  - **NDCG@3 (ความถูกต้องของการจัดอันดับ)**: `0.9453` (แบบ Hybrid) และ `1.0000` (แบบ BM25)

---

## 3. เจาะลึกหน้าที่ของทุกไฟล์ (File-by-File Details)

### 3.1 โครงสร้างโฟลเดอร์หลัก
* **`build_index.py`**:
  * **หน้าที่**: สคริปต์สำหรับเตรียมข้อมูลและสร้างดัชนีการค้นหา (Index Builder) รันครั้งเดียวเมื่อข้อมูลเปลี่ยน
  * **ขั้นตอนย่อยที่มันทำ**:
    1. อ่านข้อมูลจาก `data/astronomy_knowledge_base.txt` (ด้วย `load_qa_file`)
    2. ตัดข้อความเป็นชิ้นย่อยขนาด 400 ตัวอักษร (ด้วย `build_chunks`)
    3. นำข้อความไปเจนเวกเตอร์ขนาด 384 มิติ ด้วยโมเดล `paraphrase-multilingual-MiniLM-L12-v2`
    4. บันทึก Index เวกเตอร์ของ FAISS ไว้ที่ `vector_db/document.index`
    5. สร้างดัชนีคีย์เวิร์ดของ BM25 บันทึกไว้ที่ `vector_db/bm25_index.pkl`

* **`config.py`**:
  * **หน้าที่**: ควบคุมสวิตช์เปิด-ปิด ฟีเจอร์ทั้งหมดและที่อยู่ไฟล์ เช่น `USE_HYBRID`, `USE_RERANK`, `USE_MEMORY`, และตั้งค่าโมเดล LLM ปลายทาง

* **`main.py`**:
  * **หน้าที่**: จุดเชื่อมต่อกับผู้ใช้ (User Interface) ในรูปแบบ CLI คอมมานด์ไลน์อินเตอร์แอคทีฟ

### 3.2 ซอร์สโค้ดในโฟลเดอร์ `src/` (แกนหลักของ RAG)

* **[`src/document_loader.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/document_loader.py)**:
  * **หน้าที่**: ทำหน้าที่โหลดไฟล์ดิบ `.txt` แปลงให้อยู่ในรูปโครงสร้างข้อมูล JSON โดยใช้สัญลักษณ์ประจำหมวด `[หมวด: ...]` และแยกคำถาม (`Q:`) คำตอบ (`A:`) พร้อมผูกเลขบรรทัดเดิมไว้สำหรับใช้ทำอ้างอิงแหล่งที่มา

* **[`src/text_splitter.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/text_splitter.py)**:
  * **หน้าที่**: ซอยข้อความตัวอักษรขนาดยาวออกเป็นชิ้นเล็กๆ (Chunk) ขนาด 400 ตัวอักษร โดยมีช่วงเหลื่อมซ้อนกัน 50 ตัวอักษร เพื่อไม่ให้ข้อความส่วนปลายขาดตอนออกจากกัน

* **[`src/embedding_model.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/embedding_model.py)**:
  * **หน้าที่**: ห่อหุ้มคลาส `SentenceTransformer` ของ Hugging Face เพื่อแปลงข้อความเป็นตัวเลขเวกเตอร์มิติสูง และทำการปรับขนาดเวกเตอร์ให้มีความยาวเท่ากับ 1 (Normalize Embeddings) เพื่อให้เปรียบเทียบระยะห่างในฐานข้อมูลได้เร็วขึ้น

* **[`src/vector_store.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/vector_store.py)**:
  * **หน้าที่**: ตัวควบคุมฐานข้อมูลเวกเตอร์ FAISS โดยสร้างดัชนีแบบ `IndexFlatIP` (Inner Product) ซึ่งเมื่อใช้ร่วมกับเวกเตอร์ที่ Normalize แล้ว ค่า Inner Product จะมีค่าเทียบเท่ากับ Cosine Similarity ทำให้ค้นหาความใกล้เคียงเชิงความหมายได้ถูกต้องที่สุด

* **[`src/query_transform.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/query_transform.py)**:
  * **หน้าที่**: แก้ปัญหาคีย์เวิร์ดไม่ตรงกับเอกสารจริง ด้วยการปรับเปลี่ยนและเสริมคำถามของผู้ใช้ แบ่งเป็น:
    1. *ระดับไม่ใช้ AI*: ตัดคำลงท้ายที่ไม่ช่วยในการค้นหา (ครับ/ค่ะ/คะ) และแปลงคำแสลงในดาราศาสตร์ (ถ้ามี)
    2. *ระดับใช้ AI (LLM)*:
       - **Rewrite**: เขียนคำถามสั้นให้ยาวและชัดเจนขึ้นโดยใช้ประวัติคุยเก่ามาขยายความ
       - **Multi-Query**: แตกเป็นคำถามที่มีเป้าหมายเดียวกันแต่ใช้คำต่างกัน 3 คำถาม เพื่อค้นหาข้อมูลให้เจอครอบคลุมยิ่งขึ้น
       - **HyDE (Hypothetical Document Embeddings)**: สั่งให้ LLM ลองเขียนคำตอบสมมติขึ้นมาก่อน แล้วนำโครงสร้างคำตอบสมมตินั้นไปค้นหาหาบทความจริงในระบบ (เพราะคำตอบกับคำตอบจะมีเวกเตอร์ที่ใกล้กันมากกว่าคำถามกับคำตอบ)

* **[`src/hybrid_retriever.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/hybrid_retriever.py)**:
  * **หน้าที่**: ผสานระบบค้นหา 2 ขั้วเข้าด้วยกัน:
    1. **Dense Search (ค้นด้วยความหมาย)**: ค้นหาความใกล้เคียงผ่าน FAISS
    2. **Sparse Search (ค้นด้วยคีย์เวิร์ด)**: ค้นคำเฉพาะตัวเลขหรือทับศัพท์ภาษาอังกฤษด้วย `rank_bm25` (ผ่านกลไกตัดคำไทยด้วย PyThaiNLP `newmm`)
  * **การรวมผล RRF**: นำลิสต์ผลลัพธ์อันดับจากทั้ง 2 วิธีมาคำนวณคะแนนใหม่โดยใช้สูตร $\frac{1}{k + \text{Rank}}$

* **[`src/rerankers.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/rerankers.py)**:
  * **หน้าที่**: นำเอกสารเข้ารอบสุดท้าย 20 ชิ้น มาส่งประเมินกับโมเดล Reranker (ระดับ Cross-Encoder) เพื่อให้คะแนนความตรงประเด็นแบบคู่ต่อคู่ (Query-Doc Pair) เพื่อคัดเอาคำตอบที่ดีที่สุด 3 ชิ้นส่งให้ LLM ตอบคำถาม

* **[`src/memory.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/memory.py)**:
  * **หน้าที่**: ระบบจดจำประวัติสนทนาในอดีต (Memory) โดยจะจำจำกัดที่จำนวน `MEMORY_MAX_TURNS` เพื่อประหยัด Token และป้องกันไม่ให้ระบบสับสนเนื้อหา

* **[`src/prompt_templates.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/prompt_templates.py) & [`src/generator.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/generator.py)**:
  * **หน้าที่**: จัดทำโครงสร้าง Prompt และเรียกใช้งานโมเดลภาษา (LLM) ในการสรุปและเรียบเรียงคำตอบขั้นสุดท้าย โดยบังคับให้ใส่แหล่งอ้างอิง `[1]`, `[2]` ยืนยันว่าไม่ได้เดาข้อมูลเอง

---

## 4. โฟลว์การทำงานแบบเจาะลึก (Step-by-Step Execution Flow)

หากนำเสนอ 1:1 กับอาจารย์ ให้ไล่สายตาและอธิบายขั้นตอนการทำงานจริงของระบบตามฟังก์ชัน `RAGPipeline.ask()` ใน [`src/rag_pipeline.py`](file:///Users/cheewakornartdeelang/Desktop/cpe691/Advanced_AI/Advanced-Topic-in-Computer-Software-Course/LAB3/04-RAG-Project/src/rag_pipeline.py#L42-L83) ดังนี้ครับ:

```text
[ผู้ใช้ป้อนคำถาม] -> Q: "ทำไมดาวอังคารมีสีแดง"
  ↓
[Step 1: โหลดประวัติสนทนา]
  - ดึงข้อมูลบริบทจาก ConversationMemory (src/memory.py) 
  - เช็กว่าคำถามนี้เป็นคำถามสืบเนื่องหรือไม่ด้วย is_followup()
  ↓
[Step 2: ปรับแต่งคำถาม (Query Transform)]
  - ส่งคำถามเข้าสู่ QueryTransformer.transform() (src/query_transform.py)
  - ในกรณี Multi-Query จะสร้างคำถามเพิ่มขึ้นมาเป็น:
    1. "ทำไมดาวอังคารมีสีแดง" (Verbatim)
    2. "สาเหตุที่ดาวอังคารเป็นสีส้มแดงเกิดจากอะไร" (Alternative 1)
    3. "องค์ประกอบพื้นผิวดาวอังคารทำไมมีสีแดง" (Alternative 2)
  ↓
[Step 3: ค้นหาข้อมูลเชิงลึก (Hybrid Retrieval)]
  - เรียกใช้ retriever.retrieve() (src/hybrid_retriever.py) 
  - ยิงคำถามทั้งหมดไปค้นหาพร้อมกันสองที่:
    - ค้นหาด้วยความหมาย (Semantic Search) ใน FAISS Index
    - ค้นหาด้วยข้อความคำหลัก (Keyword Search) ใน BM25 Index
  - นำผลลัพธ์อันดับจากทุกรอบมารวมคะแนนและขจัดความซ้ำซ้อนด้วย Reciprocal Rank Fusion (RRF)
  ↓
[Step 4: คัดกรองใหม่ (Reranking)]
  - ส่งเอกสารที่เข้ารอบ 20 ชิ้นไปให้ Cross-Encoder (src/rerankers.py) ตรวจสอบความถูกต้องอย่างละเอียด
  - เลือกเอกสารที่ดีที่สุด 3 ชิ้นกลับคืนมา (TOP_K = 3)
  ↓
[Step 5: สร้าง Prompt และเรียก LLM เขียนคำตอบ]
  - จัดเตรียมข้อมูล Context ด้วย format_context() (src/prompt_templates.py)
  - นำคำถามและ Context มัดรวมใน Prompt ของระบบ (SYSTEM_PROMPT)
  - ส่งคำสั่งไปยัง Generator.generate() (src/generator.py) เพื่อเรียกใช้ LLM (Ollama/Gemini/OpenAI) เขียนเรียบเรียงคำตอบเป็นภาษาไทย
  ↓
[Step 6: จัดเก็บเข้าหน่วยความจำและตอบผู้ใช้]
  - นำคำตอบบันทึกเก็บเข้า ConversationMemory (เพื่อใช้เป็นประวัติในคำถามรอบถัดไป)
  - คืนคำตอบให้แก่ผู้ใช้ พร้อมทั้งแสดงแหล่งอ้างอิงพิกเซลและคะแนนใน Debug
```

---

## 5. คำถามยอดฮิตที่อาจารย์อาจจะถามตอนนำเสนอแบบ 1:1

**Q: ทำไมต้องใช้ Hybrid Search? ใช้แค่เวกเตอร์ค้นหา (Dense Search) อย่างเดียวไม่ดีกว่าหรอ?**  
> *แนวคำตอบ:* การใช้เวกเตอร์ค้นหา (Dense Search) อย่างเดียวมีจุดอ่อนคือ ไม่สามารถค้นหาคำเฉพาะเจาะจงที่เป็นชื่อเฉพาะ เช่น ตัวเลขทางคณิตศาสตร์, รหัสมหาสมุทร, ชื่อกล้องอวกาศเฉพาะทาง (เช่น JWST, CMB, LIGO) ได้ดีเท่าที่ควร เพราะเวกเตอร์ของคำสั้นๆ เหล่านี้มักถูกกลืนหายไปในประโยค การเพิ่ม BM25 เข้ามาจึงช่วยให้ระบบดึงเอกสารที่มีคำสำคัญเหล่านี้ได้อย่างแม่นยำ 100% ครับ

**Q: Reciprocal Rank Fusion (RRF) ทำหน้าที่อะไรในระบบนี้ และใช้สูตรยังไง?**  
> *แนวคำตอบ:* RRF คือวิธีรวมผลการจัดอันดับจากแหล่งข้อมูลค้นหาที่ต่างระบบกัน (FAISS และ BM25) เนื่องจากคะแนนดิบของ FAISS (เป็น Cosine Distance) และคะแนนของ BM25 (เป็นสถิติความถี่คำ) ไม่สามารถนำมาบวกกันตรงๆ ได้ RRF จึงใช้ความสัมพันธ์ทางคณิตศาสตร์จาก **"อันดับที่ค้นเจอ"** มาคิดคะแนนแทนด้วยสูตร:
> $$\text{Score}_{\text{RRF}}(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
> โดย $r_m(d)$ คืออันดับของเอกสาร $d$ ในระบบการค้นหา $m$ และค่า $k$ คือค่าคงที่ป้องกันคะแนนล้น (ค่าเริ่มต้นในระบบนี้คือ 60) ครับ

**Q: การประเมินคุณภาพ (Evaluation) ในระบบนี้ใช้อะไรวัดผล และดูอย่างไร?**  
> *แนวคำตอบ:* เราใช้ **Golden Set** ที่สร้างจากข้อสอบดาราศาสตร์จำนวน 27 ข้อในการวัดผลการค้นหา และมีเกณฑ์ชี้วัดหลัก 2 ตัวคือ:
> 1. **Hit@10**: ดูว่าเอกสารที่ถูกต้องติด 1 ใน 10 อันดับแรกที่ระบบค้นหามาหรือไม่ (ได้คะแนนเต็ม 1.0000 หรือ 100%)
> 2. **MRR (Mean Reciprocal Rank)**: ดูว่าเอกสารคำตอบที่ถูกต้องที่สุดอยู่ใกล้อันดับแรกสุดแค่ไหน ซึ่งระบบทำคะแนนได้เฉลี่ย 1.0000 แสดงว่าเอกสารตอบคำถามส่วนใหญ่ถูกดึงมาอยู่ในอันดับที่ 1 เสมอครับ
