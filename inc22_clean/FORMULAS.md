# Công thức toán học trong `inc22_clean` — giải thích chi tiết

Tài liệu này liệt kê **mọi công thức** dùng trong model `SetTransformerMixture`
(`inc22_fixed_aslot`), trong hàm mất mát (loss) và trong bước decode. Mỗi mục gồm:
**công thức → ý nghĩa → vị trí trong code**.

Quy ước ký hiệu:

| Ký hiệu | Ý nghĩa |
|---|---|
| $B$ | batch size |
| $N$ | số peak (token) tối đa trong 1 mẫu |
| $d$ | `d_model` = 128 |
| $K$ | số donor-slot của panel = `n_classes` = 45 |
| $M$ | số inducing point của ISAB = 32 |
| $\sigma(\cdot)$ | hàm sigmoid logistic |
| $\odot$ | nhân theo phần tử (element-wise) |
| $\mathbf 1[\cdot]$ | hàm chỉ thị (indicator) |

---

## A. Token embedding (đầu vào)

### A.1 Periodic PLR embedding (Gorishniy 2022)

Mỗi đặc trưng số $x_j$ (đã chuẩn hoá) được nhúng bằng một bộ tần số học được rồi qua Linear riêng:

$$
\mathrm{per}(x_j) = \big[\sin(2\pi\, c_{j,1} x_j),\dots,\sin(2\pi\, c_{j,F} x_j),\;
\cos(2\pi\, c_{j,1} x_j),\dots,\cos(2\pi\, c_{j,F} x_j)\big]
$$

$$
e_j = \mathrm{ReLU}\!\big(W_j\,\mathrm{per}(x_j) + b_j\big), \qquad
c_{j,k}\sim\mathcal N(0,\sigma^2),\;\sigma=0.3,\;F=8
$$

- **Ý nghĩa:** đưa 1 scalar thành vector chu kỳ (Fourier features) → mạng phân biệt được các giá trị gần nhau, tránh "nút cổ chai" khi nhồi scalar thô vào một Linear chung. Tần số $c_{j,k}$ là tham số học được.
- **Code:** `PeriodicNumEmbedding.forward` — [models/set_transformer.py:68](models/set_transformer.py).

### A.2 Chuẩn hoá đặc trưng (z-score theo tập train)

$$
\tilde x_j = \frac{x_j - \mu_j}{\sigma_j + 10^{-6}}
$$

với $\mu_j,\sigma_j$ tính từ các peak hợp lệ của tập train (`feat_mean`, `feat_std`). Code: [models/set_transformer.py:386](models/set_transformer.py).

### A.3 Phép chiếu token

$$
x = W_{\text{in}}\,\big[\,\mathrm{Emb}_{\text{locus}}(\ell)\;\Vert\;e_{1:7}\,\big]\in\mathbb R^{d},
\qquad x \leftarrow x\odot \mathrm{mask}
$$

Nối embedding locus (16 chiều) với 7×8 = 56 chiều numeric → Linear → $d=128$. Code: [models/set_transformer.py:389](models/set_transformer.py).

### A.4 `feas_filter` (lọc peak không khả thi)

$$
\text{feas}_p = \Big[\textstyle\sum_{c} \mathrm{owner\_lut}[\ell_p, b_p, c] > 0\Big],\qquad
b_p = \mathrm{round}(10\,a_p) + 30
$$

- Peak nào **không** nằm trong kiểu gen của bất kỳ donor nào (0-carrier) bị loại khỏi pad_mask trước encoder. $a_p$ = allele, $b_p$ = chỉ số bin allele.
- Code: [models/set_transformer.py:392](models/set_transformer.py).

---

## B. SetNorm (Zhang 2022)

Chuẩn hoá toàn bộ tập bằng **một** trung bình/phương sai vô hướng trên *tất cả* phần tử × đặc trưng hợp lệ, rồi affine theo từng chiều:

$$
\mu=\frac{1}{|V|\,d}\sum_{p\in V}\sum_{k=1}^{d} x_{p,k},\qquad
\mathrm{var}=\frac{1}{|V|\,d}\sum_{p\in V}\sum_{k=1}^{d}(x_{p,k}-\mu)^2
$$

$$
\mathrm{SetNorm}(x) = \frac{x-\mu}{\sqrt{\mathrm{var}+\epsilon}}\odot\gamma + \beta
$$

- $V$ = tập peak hợp lệ (theo mask). Khác LayerNorm ở chỗ thống kê lấy trên **cả tập**, giữ được thông tin tương quan độ cao giữa các peak.
- Code: `SetNorm.forward` — [models/set_transformer.py:85](models/set_transformer.py).

---

## C. Khối attention (Set Transformer++)

### C.1 MABpp — Multihead Attention Block, pre-norm, clean residual

$$
H = X + \mathrm{Attn}\big(\mathrm{SetNorm}(X),\,\mathrm{SetNorm}(Y),\,Y\big)
$$
$$
\mathrm{MABpp}(X,Y) = H + \mathrm{FFN}\big(\mathrm{SetNorm}(H)\big)
$$

trong đó attention multi-head chuẩn (softmax):

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_h}}\Big)V,\qquad d_h=d/\text{heads}
$$

FFN là `Linear(d→4d) → ReLU → Linear(4d→d)`.
- **Mask:** key bị che được cộng $-10^4$ trước softmax (BERT-style) để hàng toàn-che hoá đều $1/N$ thay vì NaN.
- Code: `MABpp.forward` — [models/set_transformer.py:112](models/set_transformer.py).

### C.2 SigmoidMABpp — attention SIGMOID **không cạnh tranh** (Ramapuram 2024)

Thay softmax bằng sigmoid theo từng phần tử, **không** chuẩn hoá theo hàng:

$$
A_{ij} = \sigma\!\Big(\frac{q_i\cdot k_j}{\sqrt{d_h}} + b\Big),\qquad
b = -\log(n_{\text{keys}})
$$

$$
\mathrm{out}_i = W_o\sum_j A_{ij}\,v_j,\qquad
H=X+\mathrm{out},\quad \mathrm{SigmoidMABpp}=H+\mathrm{FFN}(\mathrm{SetNorm}(H))
$$

- **Tại sao:** softmax buộc các key "tranh nhau" tổng = 1 → một peak MINOR mờ bị peak MAJOR cao "nuốt". Sigmoid cho mỗi cặp (query, key) một cổng độc lập ⇒ peak minor mờ vẫn được nạp vào inducing point bất kể peak major cao bao nhiêu. Bias $b=-\log(n_{keys})$ giữ tổng kỳ vọng ổn định theo kích thước tập.
- Code: `SigmoidMABpp.forward` — [models/set_transformer.py:141](models/set_transformer.py).

### C.3 ISAB++ — Induced Set Attention Block

Với $M=32$ inducing point học được $I$:

$$
H_{\text{ind}} = \mathrm{SigmoidMABpp}(I, X)\in\mathbb R^{B\times M\times d}\quad(\text{mab0, không cạnh tranh})
$$
$$
\mathrm{ISAB}(X) = \mathrm{MABpp}(X, H_{\text{ind}})\in\mathbb R^{B\times N\times d}\quad(\text{mab1, softmax})
$$

- Giảm độ phức tạp attention từ $O(N^2)$ xuống $O(NM)$. Bước 1: inducing point "đọc" cả tập bằng sigmoid (giữ minor); bước 2: từng peak đọc lại bản tóm tắt $H_{\text{ind}}$.
- Code: `ISABpp.forward` — [models/set_transformer.py:174](models/set_transformer.py).

### C.4 set_of_set (tách private/shared)

$$
n^{\text{car}}_p=\textstyle\sum_c \mathrm{owner\_lut}[\ell_p,b_p,c],\quad
\text{priv}=\{p: n^{\text{car}}_p=1\},\quad \text{shar}=\{p: n^{\text{car}}_p\ne1\}
$$
$$
H = \underbrace{\mathrm{Enc}(X;\ \text{priv})\odot\mathbf 1[\text{priv}]}_{H_{\text{private}}}
  + \underbrace{\mathrm{Enc}(X;\ \text{shar})\odot\mathbf 1[\text{shar}]}_{H_{\text{shared}}}
$$

- Peak **private** (chỉ 1 donor mang) và **shared** (≥2 donor, hoặc 0 = lạ) đi qua **cùng** encoder nhưng **riêng** mask, rồi cộng lại (hai tập rời nhau ⇒ chọn lọc sạch). Đây là tín hiệu mạnh để tách minor.
- Code: `_encode_set` — [models/set_transformer.py:413](models/set_transformer.py).

---

## D. CoSA — khởi tạo slot theo kiểu gen donor

Mỗi donor $c$ có tập allele tham chiếu; mã hoá qua **cùng** `locus_embed`+`input_proj` (các đặc trưng numeric phi-allele đặt = 0), lấy trung bình rồi chiếu:

$$
g_c = W_{\text{geno}}\left(\frac{\sum_{j} m_{c,j}\,x_{c,j}}{\sum_j m_{c,j}}\right),\qquad
W_{\text{geno}},b_{\text{geno}}\ \text{khởi tạo} = 0
$$

- $g_c$ = "neo" (anchor) cho slot của donor $c$. **Zero-init** ⇒ lúc bắt đầu $g_c=0$ (không thiên lệch), học dần.
- Code: `_encode_geno` — [models/set_transformer.py:400](models/set_transformer.py).

---

## E. Decoder — AdaptiveSlotDecoder (CoSA → GSANet → MESH → AdaSlot)

Slot ban đầu: $S^{(0)} = \texttt{geno\_slots}$ (mục D).

### E.1 GSANet — tinh chỉnh slot bằng attribution per-peak

$$
\alpha = \mathrm{softmax}(\texttt{attr\_logits})[\,:K\,]\ \odot\ \mathbf 1[\text{valid}]
\quad(\text{bỏ cột background})
$$
$$
\mathrm{agg} = W_{\text{proj}}\big(\alpha^\top H\big)\in\mathbb R^{B\times K\times d}
$$
$$
g = \sigma\!\big(W_{\text{gate}}\,\mathrm{agg}\big),\qquad
S \leftarrow S + g\odot \mathrm{agg}
$$

- $\alpha_{p,c}$ = trọng số peak $p$ thuộc donor $c$ (từ head attribution). $\alpha^\top H$ gom các peak có trọng số về từng slot; cổng $g$ kiểm soát lượng thông tin thêm vào slot.
- Code: bước 2 trong `AdaptiveSlotDecoder.forward` — [models/set_transformer.py:278](models/set_transformer.py).

### E.2 MESH — slot-attention bằng Sinkhorn-OT (lặp 3 lần)

Mỗi vòng lặp $t=1..3$:

**(i) Affinity peak↔slot:**
$$
Q = W_q\,\mathrm{LN}(S),\quad K_h=W_k H,\quad V_h=W_v H,\qquad
\text{aff} = \frac{Q K_h^\top}{\sqrt d}\in\mathbb R^{B\times K\times N}
$$

**(ii) Sinkhorn log-domain** (chuẩn hoá hai phía → ma trận xấp xỉ song ngẫu nhiên, lặp 5 lần):
$$
L \leftarrow \frac{\text{aff}}{\varepsilon},\quad \varepsilon=0.05
$$
$$
\text{lặp 5×:}\quad
L \leftarrow L - \operatorname*{logsumexp}_{K}(L)\ \ (\text{chuẩn theo slot}),\qquad
L \leftarrow L - \operatorname*{logsumexp}_{N}(L)\ \ (\text{chuẩn theo peak})
$$
$$
A = \exp(L)\in[0,1]^{B\times K\times N}
$$

**(iii) Cập nhật slot bằng GRU + FFN:**
$$
U = A\,V_h,\qquad
S \leftarrow \mathrm{GRUCell}(U,\,S),\qquad
S \leftarrow S + \mathrm{FFN}(\mathrm{LN}(S))
$$

- **Tại sao Sinkhorn (OT):** slot-attention thường softmax theo slot → một peak shared bị một slot "giành hết" (explaining-away). Chuẩn hoá **hai phía** buộc mỗi peak shared chia tỷ lệ cho nhiều slot ⇒ chống giành-hết, phù hợp peak dùng chung giữa nhiều donor. $\varepsilon$ nhỏ ⇒ gần phép gán cứng; $\varepsilon$ lớn ⇒ mềm hơn.
- Code: `sinkhorn_log` — [models/set_transformer.py:220](models/set_transformer.py); vòng lặp — [models/set_transformer.py:291](models/set_transformer.py).

### E.3 AdaSlot — cổng tồn tại slot (Gumbel-Sigmoid / Binary Concrete)

$$
\text{gate\_logit} = W_{\text{gate}}\,S\in\mathbb R^{B\times K}
$$

**Khi train** (nhiễu Logistic = hiệu hai Gumbel — relaxation của Bernoulli):
$$
u\sim\mathcal U(0,1),\qquad
L_{\text{noise}} = \log u - \log(1-u),\qquad
\text{gate} = \sigma\!\Big(\frac{\text{gate\_logit}+L_{\text{noise}}}{\tau}\Big),\ \tau=1
$$

**Khi eval** (bỏ nhiễu):
$$
\text{gate} = \sigma(\text{gate\_logit})
$$

- Cổng $\in(0,1)$ cho biết slot/donor đó "có tồn tại" trong hỗn hợp không. Nhiễu **Logistic** (không phải một Gumbel đơn) là relaxation đúng của biến Bernoulli — xem ghi nhớ *Gumbel-Sigmoid / Binary Concrete*.
- Code: bước 4 — [models/set_transformer.py:299](models/set_transformer.py).

### E.4 Logit phân loại & logit đếm

$$
\boxed{\ \text{logits\_cls} = \underbrace{\mathrm{cls\_head}(S)}_{\text{nội dung}} \;+\; \underbrace{\text{gate\_logit}}_{\text{tồn tại}}\ }
$$
$$
\text{logits\_card} = W_{\text{noc}}\,\text{gate}\in\mathbb R^{B\times 5}
$$

- Cộng trong **không gian logit** = tích hai xác suất (content × existence). `cls_head` = `LayerNorm → Linear(d→1)`.
- Code: [models/set_transformer.py:308](models/set_transformer.py).

---

## F. Các head phụ (pooling)

### F.1 PMA — Pooling by Multihead Attention

$$
Y=[\,\mathrm{rff}(X)\ \Vert\ \text{null\_kv}\,],\qquad
z = \mathrm{MAB}_{\text{softmax}}(S_{\text{seed}},\,Y)
$$

- Một "null key/value" không bao giờ bị che ⇒ tập rỗng (sau feas_filter) gộp về null thay vì NaN. $S_{\text{seed}}$ là 1 seed học được.
- Code: `PMA.forward` — [models/set_transformer.py:190](models/set_transformer.py).

### F.2 phi (độ phong phú hỗn hợp)

$$
\phi = \mathrm{softplus}\big(W_\phi\,z\big)\in\mathbb R_{\ge0}^{B\times 45},\qquad
\mathrm{softplus}(x)=\log(1+e^x)
$$

- Đầu ra không âm, hồi quy về tỷ lệ đóng góp $\phi$ của từng donor. Code: [models/set_transformer.py:466](models/set_transformer.py).

### F.3 logit_reject (open-set)

$$
z_{\text{rej}} = \mathrm{PMA}_{\text{reject}}\big(\mathrm{Enc}(X)^{\text{detach}};\ \text{không feas\_filter}\big),\qquad
\text{logit\_reject}=W_r\,z_{\text{rej}}
$$

- Encode **không lọc** (giữ bằng chứng ngoài-panel) và **detach** trước pooling ⇒ gradient reject không chạm encoder/cls. Code: `_reject_pool` — [models/set_transformer.py:437](models/set_transformer.py).

---

## G. Hàm mất mát (training)

### G.1 Asymmetric Loss — ASL (Ben-Baruch 2020) cho `logits_cls`

Với $p=\sigma(\text{logit})$, nhãn $y\in\{0,1\}$, clip $m=0.05$:

$$
p_{-} = \min(1-p+m,\ 1)\quad(\text{probability shifting cho âm})
$$
$$
\mathcal L_{\text{ASL}} = -\Big[\,y\,(1-p)^{\gamma_+}\log p \;+\; (1-y)\,p_{-}^{\,\gamma_-}\log p_{-}\,\Big]
$$

với $\gamma_+=0,\ \gamma_-=4$.

- **Ý nghĩa:** $(1-p_t)^\gamma$ là trọng số focal; $\gamma_-=4$ **hạ mạnh** các negative dễ (rất nhiều donor "vắng mặt"), tập trung vào positive khó. `clip` bỏ qua negative quá dễ. Giải bài toán mất cân bằng (45 lớp, chỉ vài lớp dương).
- Code: `AsymmetricLoss` — [train_set_transformer.py:87](train_set_transformer.py).

### G.2 Mục tiêu đếm NOC — EM-optimal $k$

Với mỗi mẫu, chọn $k$ tối ưu hoá đánh đổi miss/extra + phạt độ phức tạp:

$$
k^\* = \arg\min_{k\in\{1..5\}}\
\underbrace{\frac{\sum_c y_c(1-\text{top}_k)_c}{\max(\sum_c y_c,1)}}_{\text{tỷ lệ bỏ sót}}
+\underbrace{\frac{\sum_c(1-y_c)\,\text{top}_k{}_c}{k}}_{\text{tỷ lệ dư}}
+\ \lambda k,\quad \lambda=0.02
$$

trong đó $\text{top}_k$ = chỉ thị top-$k$ donor theo xác suất dự đoán.
- Code: `cardinality_target` — [train_set_transformer.py:109](train_set_transformer.py); cùng công thức trong `posthoc_cardinality` — [train_set_transformer.py:140](train_set_transformer.py).

### G.3 Cross-entropy đếm (có trọng số lớp)

$$
\mathcal L_{\text{noc}} = \mathrm{CE}\big(\text{logits\_card},\ k^\*\big),\qquad
w_j \propto \frac{1}{\#\{k^\*=j\}},\ \ \text{clip}\in[0.5,2.0]
$$

- Trọng số nghịch tần suất ⇒ cân bằng các mức NOC hiếm (NOC=5). Code: [train_set_transformer.py:351](train_set_transformer.py).

### G.4 Reject BCE

$$
\mathcal L_{\text{rej}} = \mathrm{BCEWithLogits}\big(\text{logit\_reject},\ y_{\text{open}}\big),\qquad
y_{\text{open}}=\begin{cases}0 & \text{mẫu closed}\\ 1 & \text{mẫu open-set}\end{cases}
$$

- Code: [train_set_transformer.py:367](train_set_transformer.py).

### G.5 soft_attr_label CE — nhãn mềm kiểu EuroForMix ($\phi\cdot CN$)

Với mỗi peak $p$, xác suất nó thuộc donor $c$ tỷ lệ với $\phi_c$ × số bản sao allele $CN_{c}$:

$$
\tilde y_{p,c} = \frac{\phi_c\, CN_{\ell_p,b_p,c}}{\sum_{c'}\phi_{c'}\,CN_{\ell_p,b_p,c'}}\quad(\text{nếu khả thi}),\qquad
\tilde y_{p,\text{bg}} = \mathbf 1[\text{không donor nào mang}]
$$
$$
\mathcal L_{\text{attr}} = -\frac{1}{|V|}\sum_{p\in V}\sum_{c}\tilde y_{p,c}\,\log \mathrm{softmax}(\text{logits\_attr}_p)_c
$$

- $CN=2$ nếu đồng hợp (homozygous), $1$ nếu dị hợp. Đây là nhãn "đặc quyền" (chỉ có với dữ liệu in-silico) gắn mỗi peak với donor — gần với mô hình EuroForMix (chia chiều cao theo $\phi\cdot CN$). Xem ghi nhớ *EuroForMix continuous model*.
- Code: [train_set_transformer.py:373](train_set_transformer.py).

### G.6 phi L1

$$
\mathcal L_\phi = \frac{1}{45}\sum_c \big|\,\phi_c - \phi_c^{\text{true}}\,\big|
$$

- Code: [train_set_transformer.py:389](train_set_transformer.py).

### G.7 Kendall — trọng số bất định đồng phương sai (Kendall 2018)

Hai loss phụ được cân bằng tự động bằng tham số $\log\text{var}$ học được:

$$
\mathcal L_{\text{aux}} =
e^{-s_{\text{attr}}}\,\mathcal L_{\text{attr}} + s_{\text{attr}}
+ e^{-s_{\phi}}\,\mathcal L_{\phi} + s_{\phi},\qquad s=\log\sigma^2
$$

- $e^{-s}=1/\sigma^2$ là trọng số (nhiệm vụ nhiễu nhiều → $\sigma^2$ lớn → trọng số nhỏ); $+s$ là số hạng phạt chống $\sigma^2\to\infty$. Không cần dò tay trọng số.
- Code: [train_set_transformer.py:390](train_set_transformer.py).

### G.8 Tổng loss

$$
\boxed{\ \mathcal L = \mathcal L_{\text{ASL}}
+ \alpha\,\mathcal L_{\text{rej}}
+ \beta\,\mathcal L_{\text{noc}}
+ \mathcal L_{\text{aux}}\ },\qquad
\alpha=0.5,\ \beta=0.3
$$

- Code: [train_set_transformer.py:369](train_set_transformer.py) (+ aux ở dòng 390).

### G.9 mask_peaks (augmentation)

Mỗi epoch, bỏ ngẫu nhiên ~15% peak **shared** (giữ peak private của minor), miễn còn ≥ 8 peak:

$$
\text{drop}_p \sim \mathrm{Bernoulli}(0.15)\ \wedge\ \text{valid}_p\ \wedge\ (n^{\text{car}}_p\ne1)
$$

- Tăng độ bền; **không** bao giờ bỏ peak đơn-carrier của minor (giữ NOC định danh được). Code: [train_set_transformer.py:339](train_set_transformer.py).

---

## H. Decode (suy luận)

### H.1 Khử chập $\phi$ bằng EM (uniform-compat) — `deconv_phi`

Cho mỗi mẫu, với các peak khớp một allele-key có carrier, độ cao quan sát:

$$
h_p = \mathrm{expm1}(\text{log-RFU}_p) = e^{\text{logRFU}_p}-1
$$

Ma trận tương thích (uniform giữa các carrier, có "thùng" background):
$$
S_{p,c} = \begin{cases}0 & c \text{ mang allele của }p\\ -2 & c=\text{bg}\\ -\infty & \text{ngược lại}\end{cases}
$$

Lặp EM ($n=10$):
$$
\textbf{E: }\ A_{p,c} = \mathrm{softmax}_c\big(S_{p,c} + \log\phi_c\big)\quad(\text{trách nhiệm của peak }p)
$$
$$
\textbf{M: }\ w_c = \sum_p A_{p,c}\,h_p,\quad
\phi \leftarrow \frac{[\,w_1,\dots,w_C,\ w_{\text{bg}}\,]}{\sum_c w_c + w_{\text{bg}}}
$$

- **Ý nghĩa:** chia cạnh tranh chiều cao peak cho các donor mang allele đó → ước lượng tỷ lệ hỗn hợp (Mx) kiểu EuroForMix (Bleka 2016). **Hoàn toàn không dùng trọng số model** ⇒ tín hiệu độc lập (điều kiện cần cho LOP ở H.2).
- Code: `phi_rerank.deconv_phi` — [phi_rerank.py:39](phi_rerank.py).

### H.2 Rerank — Logarithmic Opinion Pool / Product-of-Experts

Với mỗi mẫu (chuẩn z-score theo hàng $z(\cdot)$):

$$
\boxed{\ \text{score}_c = z(\text{logit}_c) + \alpha\, z\big(\log(\phi_c+10^{-6})\big)\ }
$$

- **Lý thuyết:** dưới Bayes + **độc lập**, kết hợp 2 chuyên gia = cộng log-xác suất = cộng logit (Genest & Zidek 1986; Hinton 2002). $\alpha$ dò trên val. Chỉ đổi **thứ hạng** (argsort) để chọn donor; **không** đổi việc đếm $k$.
- Code: `rerank_scores` — [phi_rerank.py:77](phi_rerank.py); dò $\alpha$: `tune_alpha` (tối đa oracle EM top-true-$k$ trên các strata NOC cao) — [phi_rerank.py:86](phi_rerank.py).

### H.3 Đếm NOC — RandomForest hậu nghiệm trên hồ sơ xác suất

Đặc trưng cho mỗi mẫu (từ $P=\sigma(\text{logits\_cls})$):

$$
\mathrm{feat}(P) = \big[\underbrace{P_{(1)},\dots,P_{(8)}}_{\text{8 xác suất lớn nhất}},\ \textstyle\sum_c P_c,\ \#\{P_c\ge 0.5\}\big]
$$

$$
\hat k = \mathrm{RandomForest}_{300,\,d=6}\big(\mathrm{feat}(P)\big),\quad
\text{huấn luyện trên val với nhãn } k^\* \text{(G.2)}
$$

- Đếm trên **xác suất** (không trên điểm đã rerank): "đếm-trên-rerank" đánh đổi N3/N4 lấy N5 (đo được ~ −13pp N3, −7pp N4) nên bị loại. Head CORN học được cũng bị loại (sập N3/N4).
- Code: `posthoc_cardinality` — [train_set_transformer.py:140](train_set_transformer.py).

### H.4 Decode top-k cuối

$$
\hat y_{i,c} = \mathbf 1\big[\,c\in \text{top-}\hat k_i\ \text{theo score (đã rerank)}\,\big],\qquad
\hat k_i=\mathrm{clip}(\mathrm{round}(\hat k_i),1,5)
$$

- "oracle" (trần lý thuyết) = top-true-$k$ trên score đã rerank. Code: `topk_decode` — [train_set_transformer.py:127](train_set_transformer.py).

---

## I. Chỉ số đánh giá

**Exact Match theo NOC** (toàn bộ tập donor đúng):
$$
\mathrm{EM} = \frac{1}{|S_j|}\sum_{i\in S_j}\mathbf 1\big[\hat y_i = y_i\big],\quad S_j=\{i:\text{NOC}_i=j\}
$$

**Oracle Recall@k** (chỉ số chọn model — macro trên NOC):
$$
\mathrm{Rec@}k_i = \frac{|\text{top-}k_i \cap \{c:y_{i,c}=1\}|}{k_i},\quad k_i=\text{NOC}_i
$$

- Code: `per_noc_em` [train_set_transformer.py:156](train_set_transformer.py), `evaluate_oracle_em` [train_set_transformer.py:207](train_set_transformer.py).

---

### Tóm tắt luồng công thức (top-down)

```
tokens ─A→ x ─C(set_of_set/ISAB++)→ H ─┐
donor_geno ─D(CoSA)→ geno_slots ───────┤
                                       ▼
                       E: GSANet → MESH(Sinkhorn ×3) → AdaSlot(Gumbel-Sigmoid)
                                       ▼
            logits_cls = cls_head(S) + gate_logit ;  logits_card = noc_head(gate)
            phi = softplus(phi_head(PMA(H))) ;  logit_reject = reject_head(PMA_rej)
                                       ▼
  loss = ASL + 0.5·BCE_rej + 0.3·CE_noc + Kendall(CE_attr + L1_phi)
                                       ▼
  decode: φ-rerank (EM Mx → LOP)  → RF count (probs)  → top-k
```
