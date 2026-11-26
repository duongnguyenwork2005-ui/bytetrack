# Khóa luận ByteTrack

Bộ nguồn gồm sáu chương, phụ lục, hình minh họa và tài liệu tham khảo BibTeX. Đường dẫn tới hồ sơ thực nghiệm được nêu ở phụ lục.

Dùng XeLaTeX và BibTeX để biên dịch cục bộ. Tệp chính là `main.tex`. Khi có Times New Roman, trình biên dịch dùng phông này. Nếu thiếu, tài liệu dùng TeX Gyre Termes.

```
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

Điền tên, mã sinh viên, ngành và giảng viên hướng dẫn trong `thong-tin.tex`. Sinh viên kiểm tra lời cam đoan và lời cảm ơn trong `dau-quyen.tex`. Các phần khoa học nằm trong `chuong/`. Hai hình PDF thực nghiệm trong `hinh/` đã được tạo sẵn. Tệp `hinh/tao-bieu-do.py` có thể tạo lại chúng nếu bộ nguồn nằm trong kho mã có các kết quả đã xác minh.
