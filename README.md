# Medicagent

Trợ lý y tế hiện đại sử dụng kiến trúc đa tác tử.

## 🖼️ Tổng quan hệ thống

![Tổng quan hệ thống Medicagent](backend/assets/Medicagent.png)

## 🩺 Tính năng

- Giao diện trò chuyện y tế với nhiều tác tử (multi-agent), hỗ trợ truy xuất và tìm kiếm thông tin.
- Phân tích hình ảnh y tế
- Hỗ tương tác bằng giọng nói
- Xác minh của con người (Human Validation)

## 🔧 Công nghệ sử dụng

**Frontend:** Next.js 14, TypeScript, Tailwind CSS, shadcn/ui  
**Backend:** FastAPI, Python 3.8+, Pytorch, Langchain, LangGraph, Qdrant

| Component | Technologies |
|-----------|-------------|
| 🔹 Agent Orchestration | LangGraph |
| 🔹 Document Parsing | Docling |
| 🔹 Knowledge Storage | Qdrant Vector Database |
| 🔹 Conversation Model | Gemini (Google) |
| 🔹 Medical Imaging | Computer Vision Models:<br>• Brain Tumor: Object Detection (PyTorch)<br>• Chest X-ray: Image Classification (PyTorch)<br>• Skin Lesion: Semantic Segmentation (PyTorch) |
| 🔹 Information Searching | Tavily API |
| 🔹 Guardrails | LangChain |
| 🔹 Speech Processing | Azure Speech |

## 📌 Các tác nhân

### 🧠 Tác vụ hội thoại

Hệ thống hỗ trợ nhiều tác nhân hội thoại chuyên biệt phục vụ cho các nhu cầu  khác nhau:

- **🗨️ Conversation Agent**  
  Tác nhân hỗ trợ hội thoại tổng quát.

- **📄 Medical RAG Agent**  
  Tác nhân truy xuất thông tin y khoa từ tài liệu và tri thức:  
  • Phân tích tài liệu PDF dựa trên Docling  
  • Xử lý và nhúng nội dung định dạng Markdown  
  • Sematic chunking
  • Tìm kiếm kết hợp với cơ sở dữ liệu vector Qdrant  

- **🔍 Web Search Agent**  
  Tác nhân tìm kiếm thông tin y học từ internet:  
  • Tìm kiếm tài liệu nghiên cứu y học thông qua PubMed  
  • Tìm kiếm đa nguồn thông minh qua Tavily API

### 🧬 Tác vụ thị giác máy tính

Nhiều mô hình thị giác máy tính đã được tích hợp để hỗ trợ phân tích hình ảnh y tế chuyên sâu:

- **🧠 Brain Tumor Agent**  
  • Phân loại hình ảnh MRI não  
  • Độ chính xác: 97.56%

- **🫁 Chest X-ray Agent**  
  • Nhận diện Covid-19 từ ảnh X-quang ngực  
  • Độ chính xác: 97%

- **🦠 Skin Lesion Agent**  
  • Phân vùng tổn thương da trên hình ảnh  
  • Dice Score: 0.784

## 🚀 Bắt đầu 

### Yêu cầu cài đặt

- Node.js 18+ 
- Python 3.8+

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Backend Setup


```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```
