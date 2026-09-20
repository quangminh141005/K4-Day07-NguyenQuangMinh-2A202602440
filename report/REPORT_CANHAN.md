# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** [Nguyễn Quang Minh]  
**MSSV:** [2A202602440]  
**Nhóm:** [G11]  
**Ngày:** 2026-09-19  

> **Nộp 1 bản / sinh viên.** Phần làm việc chung của nhóm như tài liệu sử dụng, chiến lược, bộ câu hỏi đánh giá và demo được trình bày trong `REPORT_NHOM.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) — Bài tập 1.1

**Độ tương tự cosine cao có ý nghĩa gì?**

> Cosine similarity càng gần 1 thì hai vector càng có hướng giống nhau. Đối với embedding văn bản, điều này thường cho thấy hai câu hoặc hai đoạn văn có nội dung/ngữ nghĩa gần nhau, dù chúng không nhất thiết phải dùng cùng từ.

**Ví dụ có độ tương tự CAO:**

- Câu A: "Sinh viên đăng ký môn học trên hệ thống."
- Câu B: "Người học thực hiện đăng ký học phần trực tuyến."
- Giải thích: Hai câu đều nói về cùng một hoạt động là đăng ký môn/học phần thông qua hệ thống trực tuyến, chỉ khác cách dùng từ.

**Ví dụ có độ tương tự THẤP:**

- Câu A: "Sinh viên cần hoàn thành học phí đúng hạn."
- Câu B: "Máy tính sử dụng bộ nhớ RAM để lưu dữ liệu tạm thời."
- Giải thích: Hai câu thuộc hai chủ đề không liên quan: một câu về học vụ, câu còn lại về phần cứng máy tính.

**Tại sao cosine similarity thường phù hợp hơn Euclidean distance khi so sánh text embedding?**

> Với embedding văn bản, điều quan trọng thường là hướng của vector hơn là độ lớn tuyệt đối. Euclidean distance chịu ảnh hưởng bởi magnitude, trong khi cosine similarity so sánh góc giữa hai vector. Vì vậy cosine similarity thường phù hợp hơn để đánh giá mức độ gần nhau về mặt biểu diễn ngữ nghĩa.

### Bài toán tính Chunking — Bài tập 1.2

**Tài liệu dài 10,000 ký tự, `chunk_size=500`, `overlap=50`. Có bao nhiêu chunk?**

Ta có bước nhảy giữa hai chunk:

\[
500 - 50 = 450
\]

Số chunk:

\[
\left\lceil \frac{10000 - 50}{500 - 50} \right\rceil
=
\left\lceil \frac{9950}{450} \right\rceil
=
23
\]

> **Đáp án:** 23 chunks.

**Nếu overlap tăng lên 100 thì chuyện gì xảy ra?**

Khi đó bước nhảy còn:

\[
500 - 100 = 400
\]

Số chunk:

\[
\left\lceil \frac{10000 - 100}{500 - 100} \right\rceil
=
\left\lceil \frac{9900}{400} \right\rceil
=
25
\]

> Số chunk tăng từ 23 lên 25. Overlap lớn hơn giúp giữ lại nhiều ngữ cảnh ở vùng biên giữa các chunk, giảm khả năng một câu hoặc một ý quan trọng bị cắt rời hoàn toàn.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Phần này mô tả cách tôi triển khai các thành phần chính trong thư mục `src`.

### Các hàm Chunking

#### `SentenceChunker.chunk`

> Tôi chia văn bản dựa trên ranh giới câu bằng regular expression. Sau khi tách, từng câu được `.strip()` để loại bỏ khoảng trắng thừa. Tiếp theo, tôi gom tối đa `max_sentences_per_chunk` câu vào một chunk. Với input rỗng hoặc chỉ gồm khoảng trắng, hàm trả về danh sách rỗng.

Điểm tôi chú ý ở phần này là dấu câu cần được giữ lại để nội dung chunk vẫn đọc tự nhiên và không bị mất thông tin.

#### `RecursiveChunker.chunk` / `_split`

> Tôi xử lý văn bản theo nhiều mức separator, từ ranh giới lớn đến nhỏ, ví dụ đoạn văn, dòng, câu, khoảng trắng và cuối cùng là từng ký tự. Nếu một đoạn sau khi split vẫn dài hơn `chunk_size`, hàm tiếp tục gọi đệ quy với separator tiếp theo.

Sau bước chia nhỏ, tôi gom các phần liền nhau nếu tổng độ dài vẫn không vượt quá `chunk_size`. Cách này giúp tránh tạo ra quá nhiều chunk rất ngắn.

Các trường hợp đặc biệt:

- Input rỗng → trả về `[]`.
- Độ dài text đã nhỏ hơn hoặc bằng `chunk_size` → trả về một chunk.
- Không còn separator phù hợp → cắt trực tiếp theo kích thước `chunk_size`.

### Lớp `EmbeddingStore`

#### `add_documents`

> Khi thêm tài liệu, mỗi document được chuyển thành record gồm `id`, `content`, `metadata` và `embedding`. Embedding được sinh ra thông qua embedder được truyền vào store. Sau đó record được lưu trong danh sách nội bộ.

#### `search`

> Query được encode thành vector, sau đó tính cosine similarity với từng document đã lưu. Các kết quả được sắp xếp theo `score` giảm dần và chỉ lấy tối đa `top_k` kết quả đầu tiên.

Do vector trong bài được chuẩn hóa, phép tính similarity có thể được thực hiện thông qua dot product.

#### `search_with_filter`

> Tôi lọc document theo metadata trước khi tính similarity. Việc pre-filter giúp bảo đảm các document không thuộc điều kiện lọc sẽ không chiếm vị trí trong top-k rồi mới bị loại bỏ sau đó.

#### `delete_document`

> Tôi duyệt store và loại bỏ các record có `doc_id` hoặc `id` trùng với tài liệu cần xóa. Hàm trả về `True` khi có ít nhất một record được xóa, và trả về `False` nếu không tìm thấy document tương ứng.

### `KnowledgeBaseAgent.answer`

> Agent đầu tiên kiểm tra store có dữ liệu hay không. Nếu store rỗng, agent trả về thông báo phù hợp thay vì gọi LLM.

Nếu có dữ liệu, agent:

1. Nhận câu hỏi từ người dùng.
2. Truy xuất các chunk liên quan bằng `EmbeddingStore.search`.
3. Đánh số nguồn `[1]`, `[2]`, ...
4. Ghép context vào prompt.
5. Yêu cầu LLM chỉ trả lời dựa trên context đã truy xuất.
6. Trả lại câu trả lời có trích dẫn nguồn tương ứng.

Mục tiêu của cách làm này là giảm hallucination và giúp người dùng kiểm tra được câu trả lời đến từ chunk nào.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Phần điểm này phụ thuộc vào việc mã nguồn vượt qua các test được cung cấp.

### Kết quả kiểm thử

> **Thay block bên dưới bằng kết quả thật của bạn khi chạy:**
>
> ```bash
> pytest -v
> ```

```text
[PASTE YOUR PYTEST OUTPUT HERE]
```

**Số lượng bài test vượt qua:** `[xx / 42]`

> Nếu code của bạn giống implementation đã kiểm tra trong nhóm và chạy đủ toàn bộ test, có thể ghi kết quả thực tế sau khi tự chạy lệnh kiểm thử.

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Trước khi chạy chương trình, tôi dự đoán mức độ tương tự dựa trên ý nghĩa tự nhiên của các câu.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|---|---|---|---|---:|---|
| 1 | Thủ tục đăng ký học phần trực tuyến. | Quy trình đăng ký môn học qua mạng. | cao | `[score]` | `[Đúng/Sai]` |
| 2 | Học phí cần nộp trước kỳ thi. | Chi phí học tập phải thanh toán trước ngày thi. | cao | `[score]` | `[Đúng/Sai]` |
| 3 | Quy định số tín chỉ tối đa mỗi kỳ. | Mỗi học kỳ sinh viên được đăng ký nhiều nhất bao nhiêu tín chỉ? | cao | `[score]` | `[Đúng/Sai]` |
| 4 | Học bổng khuyến khích học tập kỳ này. | Thời tiết hôm nay trời nhiều mây và có mưa rào. | thấp | `[score]` | `[Đúng/Sai]` |
| 5 | Thủ tục mượn sách thư viện trường. | Quy định giữ xe máy tại bãi đỗ xe. | thấp | `[score]` | `[Đúng/Sai]` |

**Kết quả nào làm tôi bất ngờ nhất? Điều đó cho thấy điều gì về embedding?**

> Phần gây bất ngờ nhất đối với tôi là một số cặp câu có ý nghĩa khá giống nhau nhưng score từ `MockEmbedder` có thể thấp, trong khi một số câu khác chủ đề vẫn có thể nhận score dương. Sau khi xem lại cách hoạt động của mock embedding, tôi nhận ra kết quả này là hợp lý vì `MockEmbedder` chủ yếu phục vụ việc kiểm thử pipeline chứ không thực sự học nghĩa của ngôn ngữ. Vì vậy score của nó không nên được diễn giải giống embedding từ các mô hình semantic embedding thực tế.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Tôi chạy 5 câu hỏi đánh giá chung của nhóm trên implementation cá nhân.

> **Các giá trị score, top chunk và câu trả lời bên dưới cần lấy từ lần chạy thật của bạn.**

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Score | Relevant? | Câu trả lời Agent (tóm tắt) |
|---|---|---|---:|---|---|
| 1 | Sinh viên muốn hủy học phần đã đăng ký nhưng không rút học phí thì thời hạn muộn nhất là khi nào theo quy định của UEH? | `[Điền top-1 chunk]` | `[score]` | `[Có/Không]` | `[Tóm tắt]` |
| 2 | Kế hoạch đăng ký môn học theo hệ thống tín chỉ tại UTE gồm những giai đoạn nào? | `[Điền top-1 chunk]` | `[score]` | `[Có/Không]` | `[Tóm tắt]` |
| 3 | Tại sao một học phần có thể không xuất hiện trên hệ thống đăng ký môn học của HCMUT? | `[Điền top-1 chunk]` | `[score]` | `[Có/Không]` | `[Tóm tắt]` |
| 4 | Ở đợt điều chỉnh đăng ký môn học tại UEL, sinh viên có được đăng ký bổ sung môn mới không và sinh viên đóng học phí ở đâu? | `[Điền top-1 chunk]` | `[score]` | `[Có/Không]` | `[Tóm tắt]` |
| 5 | Quy định kế hoạch học tập và tổ chức đăng ký học phần *(Filter: `audience=student`)* | `[Điền top-1 chunk]` | `[score]` | `[Có/Không]` | `[Tóm tắt]` |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?**

> `[x / 5]`

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác qua demo:**

> Sau khi so sánh nhiều cách chunking, tôi nhận thấy việc chọn ranh giới chia văn bản ảnh hưởng trực tiếp đến chất lượng retrieval. Fixed-size chunking đơn giản và dễ kiểm soát, nhưng có thể cắt một ý thành hai phần. Recursive chunking giữ cấu trúc tự nhiên tốt hơn, nhưng kích thước chunk không đồng đều. Với tài liệu có heading rõ ràng, chia theo section/heading có thể hiệu quả hơn vì mỗi phần vốn đã chứa một đơn vị thông tin tương đối hoàn chỉnh.
>
> Ngoài ra, metadata filtering rất hữu ích khi dữ liệu đến từ nhiều trường hoặc nhiều loại đối tượng khác nhau. Nếu lọc đúng trước khi tính similarity, hệ thống có thể giảm số lượng document không liên quan và tăng khả năng lấy đúng nguồn cần thiết.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | `[x / 10]` |
| Hoàn thiện code (Core Implementation — tests) | `[x / 30]` |
| Dự đoán độ tương tự (Similarity Predictions) | `[x / 5]` |
| Kết quả truy xuất của tôi (Competition Results) | `[x / 10]` |
| **Tổng phần cá nhân** | **`[x / 60]`** |

---

## Ghi chú trước khi nộp

- Điền đúng họ tên, MSSV và tên nhóm.
- Chạy lại `pytest -v` và dán kết quả thật.
- Điền score similarity thật từ chương trình.
- Chạy lại 5 query đánh giá và cập nhật bảng Competition Results.
- Không dùng kết quả của thành viên khác cho các phần yêu cầu kết quả cá nhân.
