# 🌌 Advanced RAG System: Astronomy & Space Exploration Knowledge Base
ระบบถาม-ตอบอัจฉริยะด้านดาราศาสตร์และการสำรวจอวกาศ (ภาษาไทย) พัฒนาด้วยเทคนิค Advanced RAG (Retrieval-Augmented Generation)

## 📌 สถาปัตยกรรมระบบ (RAG Architecture)
```
User Input → Memory → Query Transformation → Hybrid Search (FAISS + BM25) → RRF Fusion → Context Builder → LLM Generator → Output [n]
```

## 📚 เอกสารคู่มือและการนำเสนอ (Documentation)
* 📖 [คู่มือเตรียมสอบและนำเสนอ (Presentation Guide)](docs/rag_presentation_guide.md)
* 🎤 [สคริปต์พูดนำเสนอ 1:1 กับอาจารย์ (Presentation Script)](docs/presentation_script.md)
* 📂 [คำอธิบายทุกไฟล์และเปรียบเทียบสิ่งที่แก้ไข (File-by-File Details)](docs/file_by_file_explanation.md)
* 📊 [สรุปผลการทดสอบส่วน Retrieval (Evaluation Summary)](docs/retrieval_evaluation_summary.md)

## 🚀 วิธีการติดตั้งและใช้งาน (Quick Start)

```bash
# 1. เข้าสู่โฟลเดอร์โปรเจกต์
cd LAB3/04-RAG-Project

# 2. เปิดใช้งาน Virtual Environment
source ../../.venv/bin/activate

# 3. สร้างดัชนีการค้นหา (Index) จากชุดข้อมูล
python build_index.py

# 4. รันโปรแกรมหลักเพื่อถาม-ตอบ
python main.py

# 5. รันวัดผลประสิทธิภาพการค้นหา (Evaluation)
python -m evaluation.eval_retrieval
```

## 📊 ข้อมูลคลังความรู้และผลประเมิน (Metrics)
* **คลังข้อมูล (Knowledge Base):** 104 หัวข้อ Q&A (191 Chunks)
* **Hit@10:** 100.00% (ดึงเอกสารเป้าหมายติด Top 10 ทุกข้อ)
* **Hybrid MRR:** 0.9836
* **ความเร็วเฉลี่ย:** 20.5 ms / คำถาม
