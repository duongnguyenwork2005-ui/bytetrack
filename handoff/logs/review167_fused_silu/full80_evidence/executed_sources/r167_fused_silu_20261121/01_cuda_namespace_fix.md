# Sửa compile CUDA10.2, không đổi toán tử

V1 compile thất bại vì anonymous namespace của NVCC10.2 xung đột với
anonymous namespace trong header TensorRT khi dùng global using namespace.
Giữ fused_silu.cu/compile_v1.log. V2 dùng named namespace r167precise,
đặt using bên trong và qualified call tới launch ở C test entry point.
Chưa có kernel oracle, engine, AP hoặc timing R167 tại thời điểm sửa.

Graph transform77 vẫn hợp lệ, initializer không đổi; transform.json và
nguồn v1 giữ nguyên. Build preflight sẽ khóa riêng source v2 thật và .so.
