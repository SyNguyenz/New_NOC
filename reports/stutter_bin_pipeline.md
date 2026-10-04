# Cai gi quyet dinh chieu cao cua MOT BIN STUTTER

Muc tieu: liet ke day du, theo dung thu tu pipeline, moi ham / buoc trong `gen_mixture` co the thay doi chieu cao cua
mot bin khong phai allele, kem: hang so dieu khien no, real co cho ta "dap an" doc duoc o muc bin hay khong, va phep
thu tuong ung. Dung de chan doan; khong hang so nao duoc fit tu mixture test.

Chi so dang xet: `P(chieu cao quan sat > 3 x ky vong cua luat)` tai cac bin khong phai allele, tach theo so allele
cha dang dung dong vao bin do. Do tren NOC1 (900 ho so, 88832 bin): 1 nguon real 6.8 / gen 7.0 %; 2 nguon real 4.5 /
gen 4.0 %; >=2 nguon mot cha tro >80 % real 3.9 / gen 3.4 %.

## Cac buoc, theo thu tu

| # | buoc / ham | dieu khien | real co dap an o muc bin? | phep thu |
|---|---|---|---|---|
| 1 | `_filter_contrib` -> `contrib`, `contrib_exp` | chieu cao tung allele cua tung nguoi (khuon va lan bốc thuc hien); `FILT_CV` 0.3, `HEIGHT_SHARE`, `AMOUNT_SD` | **CO** o bin allele (tong cac nguoi). Khong tach duoc theo nguoi | chep chieu cao allele cua real (layer D). Allele real khong hien phai GIU khuon, chi an o dinh cua chinh no |
| 2 | vong `art_loc` - ty le phat | `rate` cua bang + `ALLELE_RATE[d]` | KHONG (chi thay ket qua da song sot) | bat/tat `STR_ALLELE_RATE`; so ty le co dinh tren NOC1 |
| 3 | vong `art_loc` - ty le chieu cao | `lmu`, `_sl2` (do doc theo allele), `ALLELE_STUT_OFF[d]`, `STUT_LT_A/G` | KHONG tach duoc khi nhieu nguon | so phan bo log(quan sat/ky vong) o bin SACH 1 nguon |
| 4 | tang phan tan: mau -> locus -> tung dinh | `art_run_sd` (`_roff`), `ART_LOCUS_R` (`_loff`), `ART_SCAT_K`, `lsd` | KHONG | dat `ART_LOCUS_R` 0 / 0.11 / 0.5; do tuong quan trong cung locus tren NOC1 |
| 5 | chan tren moi lan bốc | `row[7]` (ty le lon nhat NOC1 tung thay), co `STR_ART_CAP` | KHONG | `STR_ART_CAP=0` |
| 6 | `_cross(d)` - tru phan tu-stutter | phan stutter roi vao chinh allele cua nguoi do (da co trong nguon) | KHONG | chua co co |
| 7 | cong don nhieu nguon | `np.add.at(art_h, tgt, ...)`: cac nguon doc lap roi cong | KHONG | tach theo so nguon (day la cho lech) |
| 8 | song sot artefact | `ART_SURV_H/P` theo chieu cao DA THUC HIEN, `_CARR_MULT`, `SHOULDER_ART`, `STUT_SMALL_Q` | KHONG | tat tung he so; `STUT_SMALL_Q=1` |
| 9 | nhieu nen | `_NL_P`, `NOISE_LOC_Q`, `NOISE_RHO`, `RUN_COUPLE`, `NOISE_GAIN`, crowding `NOISE_CR_*`, mask `DONOR_DOSAGE` | **CO mot phan**: o bin khong co ky vong stutter, dinh quan sat CHINH LA nhieu | chep nhieu cua real o cac bin khong co stutter; tat crowding |
| 10 | drop-in | `DI_PERSON`, `DI_RATE_P`, `DI_LMU_P`, `DI_HSD` | KHONG | `DI_RATE_P=0` |
| 11 | bao hoa | `SAT_RFU` | - | khong lien quan o vung 10-100 RFU |
| 12 | loc pull-up | `PULL_P` + `pull_prob`, tinh tren profile da hoan chinh | KHONG | tat `PULL_P` |
| 13 | phat pull-up | `PU_RATE/PU_RATIO` (`STR_PU_EMIT`) | **CO mot phan** qua ngu canh kenh mau khac | da do: tang them 0.04 dinh/ho so, tro |
| 14 | nguong phat hien | `AT_PC` | **CO**: bin duoi nguong thi khong thay | doi `AT_PC` |

## Cho real KHONG cho dap an

Buoc 2, 3, 4, 5, 6, 7, 8, 10, 12 khong tach duoc o muc bin: real chi cho **mot** con so moi bin, la tong cua stutter
+ nhieu + drop-in + pull-up con sot, sau khi da qua song sot, loc va nguong. Vi vay "chep dap an tung ham" chi dinh
nghia duoc cho buoc 1, 9, 13, 14; cac buoc con lai phai thu bang **knockout** (tat / doi hang so) roi doc chi so cuoi.

## Da thu, va ket qua

| buoc | phep thu | ket qua |
|---|---|---|
| 1 | chep allele cua real | NOC2 o 2 nguon khop han (3.7 -> 4.9 %, real 4.9) khi allele an giu khuon; NOC5 3 nguon chi 1.1 -> 1.3 (real 2.8) |
| 3 | phan bo o bin sach 1 nguon | real gan dung lognormal (lech chuan +0.27); **generator moi la cai khong phai lognormal** (lech chuan -3.16, do nhon +22), sd 0.408 vs real 0.266 nhung THAN khop (p05..p99 trong 0.05) |
| 3 | `ALLELE_STUT` n-1 / n+1, `STUT_LT_A` | khop tot hon tren NOC1; KHONG chuyen chi so cuoi |
| 4 | `ART_LOCUS_R` 0.11 / 0.5 | dat tuong quan NOC1 (+0.101 vs real +0.110); KHONG chuyen chi so cuoi |
| 5 | `STR_ART_CAP=0` | 3 nguon 1.1 -> 1.7 % (real 2.8); khong phai nguyen nhan |
| 10 | drop-in = mot NGUOI | dong phan lon NOC4 (+0.139 -> +0.174, real +0.189); **cai duy nhat co tac dung** |
| 13 | phat pull-up | tang 0.04 dinh/ho so; tro |
| - | stutter cua stutter (buoc 2 ap len dinh artefact) | E2/E1 chi 5-7 %, khong tang theo NOC |

## Con lai

Duoi trai cua generator o buoc 3: sd 0.408 vs real 0.266 **chi vi duoi trai**, than khop. Nghi pham theo thu tu
pipeline: chinh viec phat tu KHUON thay vi tu chieu cao cha da thuc hien (buoc 1 gap buoc 3) - khi cha ra cao hon
khuon thi ty le doc duoc thap di, dung dang duoi trai, va co lon dung bang CV 0.3 cua lan bốc Gamma
(sqrt(0.408^2 - 0.266^2) = 0.31). Day la lua chon co chu y vi no lam cha yeu hanh xu dung, nen neu sua phai giu duoc
ca hai dau.

## Ket qua quet tung buoc (NOC1, 500 ho so, allele chep tu real)

`P(quan sat > 3x ky vong)` tai bin khong phai allele. REAL: 1 nguon **6.90 %**, >=2 nguon **4.58 %**.

| cau hinh | 1 nguon | >=2 nguon |
|---|---|---|
| generator nhu hien tai | 6.85 | 3.97 |
| 2 `ALLELE_RATE` tat | 7.08 | 3.84 |
| 3 `ALLELE_STUT` tat | 6.83 | 3.92 |
| 3 stutter template thap tat | 6.71 | 3.70 |
| 4 tang locus tat | 6.77 | 3.86 |
| 4 tang locus = 1 | 6.78 | 3.94 |
| 5 chan tren tat | 6.88 | 4.00 |
| 8 `STUT_SMALL_Q=1` | 6.90 | 3.96 |
| **8 song sot artefact = 1** | **8.30** | **5.09** |
| **9 nhieu nen tat** | **1.30** | **1.10** |
| 10 drop-in tat | 6.56 | 3.63 |
| 12 loc pull-up tat | 6.88 | 3.89 |
| 14 `AT_PC=0` | 6.68 | 4.00 |

**Doc ket qua.** Chi so nay hau het **khong phai stutter**: tat nhieu nen thi no sup tu 6.85 xuong 1.30 %, nghia la
~80 % cac "dinh vuot 3x ky vong" la NHIEU NEN roi dung vao vi tri stutter, khong phai stutter cao bat thuong. Moi
hang so cua chinh co che stutter (buoc 2, 3, 4, 5) doi chi so duoi 0.3 diem - trong bien do nhieu. Chi mot buoc
chuyen manh: **song sot artefact** (`ART_SURV_P` = 1) dua 1 nguon 6.85 -> 8.30 va >=2 nguon 3.97 -> **5.09** (real
4.58), tuc no **vuot qua** real. Vay khoang cach 0.5 diem o bin nhieu nguon nam trong duong cong song sot, va la mot
khoang cach **noi suy duoc**, khong can co che moi.

Va vi dong gop chinh la nhieu nen, chi so nay khong phai thuoc do cho luat stutter. Cac ket luan truoc do trong file
nay dua tren no (ke ca "thieu duoi o bin chong lan") phai doc lai voi canh bao do.

## Nang duong cong song sot: cai gia khong dang

`ART_SURV_LIFT` nang moi xac suat song sot ve 1 mot phan. Doc ba so cung luc tren 500 ho so NOC1
(real: 92.9 dinh/ho so, ti le artefact 0.611, P(>3E) 6.90 % / 4.58 %):

| lift | dinh/ho so | ti le artefact | P(>3E) 1 nguon | >=2 nguon |
|---|---|---|---|---|
| 0.00 (hien tai) | 92.4 | 0.610 | 6.85 | 3.97 |
| 0.05 | 92.9 | 0.612 | 6.87 | 4.04 |
| 0.10 | 93.3 | 0.614 | 6.96 | 4.09 |
| 0.20 | 94.3 | 0.618 | 7.11 | 4.17 |
| 0.35 | 95.7 | 0.623 | 7.33 | 4.28 |

De dua >=2 nguon tu 3.97 len 4.58 can lift > 0.35, luc do dinh/ho so da la 95.7 (real 92.9) va ti le artefact 0.623
(real 0.611), nghia la **hai so ma duong cong nay duoc calibrate tren da lech ra ngoai**. O lift 0.05 - muc dinh/ho so
khop real chinh xac nhat (92.9/92.9) - chi so chi len 4.04. Vay 0.5 diem con lai la mot **danh doi**, khong phai mot
hang so dat sai: khong the lay no ma khong lam lech so dinh va ti le artefact.

Ket luan cuoi cua ca dot: chi so nay ~80 % la nhieu nen, buoc duy nhat dieu khien no la song sot artefact, va song sot
artefact dang o dung cho theo hai thuoc do NOC1 goc. Khong sua gi them.

## Quet tung buoc con cua NHIEU NEN (day moi la lop chiem 80 % chi so)

Real: 92.9 dinh/ho so, artefact 0.611, P(>3E) 6.90 % / 4.58 %.

| cau hinh | dinh/ho so | artefact | 1 nguon | >=2 nguon |
|---|---|---|---|---|
| nhu hien tai | 92.4 | 0.610 | 6.85 | 3.97 |
| `SHOULDER_NOISE = 1` (khong dap canh allele) | 95.8 | 0.624 | 6.99 | **3.95** |
| `SHOULDER_NOISE` x2 | 94.1 | 0.618 | 6.80 | 3.87 |
| `SHOULDER_ART = 1` | 92.5 | 0.610 | 6.86 | 3.97 |
| crowding tat | 108.5 | 0.669 | 9.10 | 5.13 |
| `NOISE_GAIN = 0` | 92.7 | 0.611 | 6.87 | 3.98 |
| `NOISE_RHO = 0` | 92.2 | 0.609 | 6.77 | 3.95 |
| `RUN_COUPLE = 0` | 92.5 | 0.610 | 6.82 | 3.98 |
| bo chia `_PBAR_NOISE` | 89.2 | 0.595 | 6.42 | 3.64 |
| tan suat nhieu x1.3 | 103.8 | 0.654 | 8.48 | **4.62** |

**Ket qua quan trong: `SHOULDER_NOISE` KHONG phai nguyen nhan.** Tat han no (khong dap nhieu canh allele) dua dinh/ho so
tu 92.4 len 95.8 nhung chi so >=2 nguon **khong nhich** (3.97 -> 3.95). Vay nhieu them do khong roi vao cac bin dang
xet. Gia thuyet "dap nhieu o vi tri stutter" sai.

Chi hai cau hinh dua >=2 nguon len tren 4.5: tat crowding (5.13) va tang tan suat nhieu 1.3 lan (4.62). Ca hai deu keo
dinh/ho so len 104-108 so voi real 92.9, tuc lech 12-17 %. Khong co cau hinh nao trong toan bo hai lan quet dat duoc
chi so cua real ma giu dinh/ho so o 92.9. **Dieu nay noi rang chi so khong the dat duoc bang cach doi do LON cua nhieu;
neu co cho sai thi la cho nhieu roi VAO DAU, va khong bien nao trong pipeline hien tai dieu khien viec do.**

## Tach stutter khoi nhieu nen bang phu thuoc vao allele cha

Stutter ti le voi cha (nen ti le voi ky vong E), nhieu nen thi khong. Chi giu cac bin ma **3E > dinh nhieu cao nhat cua
chinh ho so do**: o do nhieu mot minh KHONG THE dua bin vuot 3E, nen P(>3E) la thuoc do thuan cua duoi stutter.
NOC1, 500 ho so, 47924 bin co ky vong; nhieu nen cao nhat TV 19 RFU; 26 % bin nam trong vung sach.

| | n | real P(>3E) | gen P(>3E) |
|---|---|---|---|
| **vung SACH** 1 nguon | 8298 | 1.61 % | 1.33 % |
| **vung SACH** >=2 nguon | 4455 | **1.05 %** | **0.83 %** |
| vung lan, 1 nguon | 31153 | 8.36 % | 8.38 % |
| vung lan, >=2 nguon | 4018 | 8.71 % | 7.64 % |

**Ket luan.** Trong vung sach, duoi stutter that su thieu o CA HAI nhom - 1 nguon 1.33 vs 1.61 %, >=2 nguon 0.83 vs
1.05 % - ti le thieu deu nhau (~0.8x), khong tang theo so nguon. Trong vung lan, 1 nguon khop chinh xac (8.38 vs 8.36)
va chi >=2 nguon lech. Vay:
1. Co MOT thieu hut that o duoi stutter, nhung no **khong phu thuoc so nguon** va chi bang ~0.3 diem tren nen 1-1.6 %.
2. Phan lech "tang theo so nguon" ma ca dot nay duoi theo chi ton tai trong vung lan, tuc no la **nhieu nen tai vi tri
   stutter**, khong phai stutter. Ket luan "cang nhieu nguon cang thieu" la mot **hieu ung cua phep do**.
Con so co y nghia duy nhat con lai: duoi stutter thieu ~20 % tuong doi, deu tren moi so nguon, do duoc tren NOC1.

## Nhieu nen TAI vi tri stutter - do duoc, va no khop

Lay phia nguoc lai cua vung sach: bin LA vi tri stutter nhung ky vong nho hon 1/3 nhieu nen trung vi cua ho so, nen
dinh o do gan nhu thuan nhieu. NOC1, 500 ho so.

| nhom | n | real co% / RFU TV | gen co% / RFU TV |
|---|---|---|---|
| khong phai vi tri stutter (doi chung) | 217496 | 7.01 / 9.0 | 7.14 / 8.9 |
| vi tri stutter, 1 allele canh | 26920 | 8.60 / 9.0 | **9.32** / 8.8 |
| vi tri stutter, >=2 allele canh | 2744 | 10.09 / 9.0 | 9.84 / **8.2** |

Real co nhieu nen **cao hon** o vi tri stutter (8.60 va 10.09 % so voi 7.01 % o cho khac), va generator **cung co**
(9.32 va 9.84 so voi 7.14). Hai ben khop trong 0.7 diem; o >=2 allele canh generator con thap hon chieu cao mot chut
(8.2 vs 9.0 RFU). Vay lop nhieu **khong sai o vi tri stutter**, va gia thuyet `SHOULDER_NOISE` ap sai cho da bi loai
lan thu hai, lan nay bang phep do truc tiep chu khong phai knockout.

**Trang thai cuoi cua ca dot:** khong co buoc nao trong 14 buoc bi chung minh la sai. Da do va khop: allele, nhieu nen
(ca o cho khac va tai vi tri stutter), ti le stutter tung nguon o bin sach, hinh dang phan bo (real gan lognormal),
vi tri drop-in sau khi sua thanh "mot nguoi". Con lai: duoi stutter thieu ~20 % tuong doi deu tren moi so nguon, buoc
so huu no la song sot artefact, va nang song sot thi doi so dinh moi ho so. Gate: NOC2 khop chinh xac, NOC3 +.009,
NOC4 -.032, NOC5 -.069; chi so cuoi twin va real khop trong 0.003.

---

# Phan 2: pipeline sinh SU CO MAT cua allele

Do vi sao: thang chep dap an chay lai voi moc da sua cho thay khoang cach gate nam o lop allele - chep "allele nao
dung" dua NOC4 tu +0.149 len dung real (+0.189) va NOC5 tu +0.051 len +0.063 (real +0.103), con chep chieu cao thi vot
qua. Ket luan cu "artefact moi la nguyen nhan" la do moc dat allele an ve 0, xoa luon stutter cua chung.

Chi so cham: **lech sum(gate)** tren twin cua 1333 mixture test, theo tung NOC. Khong dung P(>3E) nua.

| # | buoc / ham | dieu khien | phep thu |
|---|---|---|---|
| 1 | chon nguon NOC1 (`build_ss_pool`, `_pick_source`) | ho so NOC1 nao duoc lay lam nguon cho tung nguoi | doi tieu chi chon; dung nguon day nhat |
| 2 | `_relottery` | boc lai xem allele nao co mat, giu trung binh | `STR_RELOTTERY=0` |
| 3 | buoc chuyen muc nguon -> muc nguoi (`_lvl_surv`, `_q`) | mat allele khi nguon giau hon nguoi gop | dat `_q = 0` |
| 4 | tach locus / phan tu (`_amp_split`, `_ushock`) | phan mat dung chung ca locus vs rieng tung allele | dat `_l = 0`; tat `_ushock` |
| 5 | phu thuoc kich thuoc (`_conv_surv`, `_spread_surv`) | allele dai mat nhieu hon | dung `_u` phang thay vi theo size |
| 6 | chuyen doi xu ly (`TREAT_CONV`, `FILT_S`) | nguoi bi xu ly lay tu thang chua xu ly | tat buoc chuyen |
| 7 | hieu suat ong (`KAPPA_MAX`) | 2^(-kappa(K-1)) tren moi ban sao | `STR_KAPPA_FIX=0` |
| 8 | sai so luong (`AMOUNT_SD`) | lech giua nhan va luong thuc | `STR_AMOUNT_SD=0` |
| 9 | Poisson/Gamma ban sao (`FILT_PG`, `FILT_CV`) | so ban sao va bien dong chieu cao | doi `FILT_CV` |
| 10 | chia chieu cao trong locus (`HEIGHT_SHARE`) | hai allele cua mot locus dung chung chieu cao | `STR_HEIGHT_SHARE=0` |
| 11 | nguong phat hien (`AT_PC`) | duoi nguong thi khong thay | doi `AT_PC` |

## Quet knockout lop su-co-mat-allele, cham bang GATE

Twin cua 1333 mixture test. REAL: +0.127 +0.159 +0.189 +0.103.

| cau hinh | NOC2 | NOC3 | NOC4 | NOC5 | dinh/mau |
|---|---|---|---|---|---|
| twin nhu hien tai | +0.111 | +0.165 | +0.165 | +0.055 | 131.1 |
| `_relottery` tat | +0.143 | +0.186 | +0.126 | +0.064 | 130.9 |
| `kappa = 0` | +0.111 | +0.165 | +0.165 | +0.055 | 131.1 |
| `AMOUNT_SD = 0` | +0.109 | +0.150 | +0.142 | +0.028 | 131.0 |
| `FILT_CV = 0.15` | +0.105 | +0.192 | +0.154 | +0.061 | 131.1 |
| `FILT_CV = 0.45` | +0.157 | +0.187 | +0.142 | +0.019 | 131.0 |
| `HEIGHT_SHARE = 0` | +0.137 | +0.166 | +0.139 | +0.046 | 130.9 |
| `AT_PC = 1` | +0.141 | +0.166 | +0.126 | +0.072 | 132.1 |
| `AT_PC = 6` | +0.098 | +0.142 | +0.162 | +0.016 | 125.0 |

**Khong buoc nao dua NOC5 len gan +0.103.** Cao nhat la `AT_PC=1` voi +0.072, nhung no keo NOC4 tu +0.165 xuong +0.126
(real +0.189) va them 1 dinh/mau. Moi cau hinh khac hoac lam NOC5 te hon hoac doi NOC4 lay NOC5.
**Va `kappa = 0` khong doi gi ca** (giong den ba chu so) - dang le no phai doi, vi kappa nhan vao moi ban sao cua moi
nguoi trong ong. Dieu do noi rang trong duong build twin hien tai kappa khong den duoc noi no phai den, hoac bi mot
buoc sau do ghi de. Day la mot dau hieu code, khong phai ket qua sinh hoc, va la thu dang kiem truoc tien.

## Sua knockout kappa, va lui len tang tren (chon nguon)

`kappa` KHONG vo hieu - bo thu cu dat `KAPPA_FIX` trong khi ham build truyen `kappa` tuong minh, nen nhanh do bi bo qua.
Truyen vao dung cach:

| cau hinh | NOC2 | NOC3 | NOC4 | NOC5 | dinh/mau |
|---|---|---|---|---|---|
| twin nhu hien tai | +0.111 | +0.165 | +0.165 | +0.055 | 131.1 |
| `kappa = 0` (truyen vao) | +0.137 | +0.164 | +0.139 | +0.057 | 125.8 |
| `kappa = 0.26` | +0.110 | +0.160 | +0.092 | **-0.067** | 123.0 |
| nguon DAY nhat cua donor | +0.082 | +0.152 | +0.166 | +0.010 | 124.6 |
| nguon NGHEO nhat | +0.080 | +0.139 | +0.149 | **-0.044** | 122.0 |
| REAL | +0.127 | +0.159 | +0.189 | **+0.103** | - |

Doc: kappa co tac dung manh va dung huong vat ly (kappa cao -> moi nguoi mo di -> NOC5 tu +0.055 xuong -0.067), nhung
kappa = 0 cung chi cho +0.057. Chon nguon day nhat lam NOC5 **te hon** (+0.010), nguon ngheo nhat cang te (-0.044).
Vay khong mot bien nao trong ca hai tang - luat su co mat VA viec chon nguon - dua NOC5 tu +0.055 len +0.103. Moi bien
deu di sai huong hoac doi NOC4 lay NOC5.

Dieu nay co nghia: **cai tao ra +0.103 o real khong phai mot tham so nao cua duong sinh hien tai**. Chep dap an cho thay
no o "allele nao dung", nhung khong luat nao trong generator dieu khien duoc dung cai do. Nen no la mot dac tinh cua
cach cac allele that PHAN BO giua cac nguoi trong mot hon hop that - thu ma chi dat duoc bang cach chep, khong bang cach
dieu chinh.

---

# Phan 3: leave-one-out dung cach, va danh sach chi so can sua

Chep dap an theo kieu **moi bin lay dung MOT nguon**: real o khap noi, TRU mot loai do generator sinh (allele duoc chep
TRONG pipeline, truoc buoc artefact, nen stutter cua generator van duoc sinh tu cha dung). Cong them dap an len tren
cai generator da sinh la **dem hai lan** - do la cho moi ket luan truoc bi lech.
Tu kiem "chep tat ca" tai tao real dung ba chu so, nen bo thu nay tin duoc.

REAL: +0.127 +0.159 +0.189 +0.103. Trung binh 3 hat giong, kem bien do:

| loai do generator sinh | NOC2 | NOC3 | NOC4 | NOC5 | bien do hat giong |
|---|---|---|---|---|---|
| **n-1** | +0.259 | +0.457 | **+0.532** | +0.230 | .024 .009 .044 .013 |
| **allele** | +0.333 | +0.482 | +0.450 | +0.284 | .040 .033 .058 .033 |
| nhieu nen | +0.147 | +0.191 | +0.203 | +0.122 | .010 .015 .007 .009 |
| stutter khac (n-2,n-3,half) | +0.142 | +0.200 | +0.207 | +0.105 | .015 .027 .010 .007 |
| n+1 | +0.118 | +0.172 | +0.200 | +0.080 | .027 .015 .041 .016 |
| allele VA n-1 cung luc | +0.108 | +0.149 | +0.134 | +0.056 | - |

## Danh sach chi so can sua, theo do lon

| uu tien | chi so | lech so voi real | ghi chu |
|---|---|---|---|
| 1 | **n-1 stutter** | NOC4 +0.343, NOC3 +0.298, NOC5 +0.127 | lon gap ~10 lan phan con lai |
| 1 | **allele (co mat + chieu cao)** | NOC2 +0.206, NOC3 +0.323, NOC4 +0.261, NOC5 +0.181 | nguoc chieu voi n-1: hai cai TRIET TIEU nhau, nen chep rieng tung tang luon trong vo can |
| 3 | nhieu nen | NOC3 +0.032, NOC5 +0.019 | that (bien do .015/.009) |
| 4 | stutter khac | NOC3 +0.041 | sat mep bien do .027 |
| 5 | n+1 | NOC5 -0.023 | nguoc chieu voi cac loai khac |

Moi loai artefact do generator sinh deu day gate len cao hon real o NOC3/NOC4; chi n+1 o NOC5 la nguoc.

## Ban do phu thuoc: leave-one-out hop le voi bien nao

Bai hoc tu cap (chieu cao allele, n-1): **leave-one-out khong tach duoc mot bien khoi nhung bien duoc tinh TU no.** Vay
phai soat ca danh sach. Doc tu code:

| bien | duoc tinh TU | => leave-one-out |
|---|---|---|
| allele (co mat) | nguon NOC1 + cac luat dropout | **hop le** - do duoc rieng, va da khop (+.127/.164/.192/.098) |
| allele (chieu cao) | nguon + budget + Gamma | **KHONG** rieng duoc: moi lop stutter sinh tu no |
| n-1, n+1, n-2, n-3, half | `parent = contrib_exp.sum(0)` = chieu cao allele | **KHONG** rieng duoc khoi chieu cao |
| song sot artefact | `art_h` da thuc hien -> tuc tu ty le va tu cha | KHONG rieng duoc khoi hai cai tren |
| nhieu nen: he so shoulder | `_shoulder_h(... mix ...)` - chieu cao allele quanh bin | KHONG rieng duoc khoi allele |
| nhieu nen: crowding | `_occ = (mix >= AT_PC).sum()` - so bin dang dung, tuc allele + artefact | KHONG rieng duoc khoi ca hai lop tren |
| nhieu nen: `NOISE_GAIN` | tong quan sat so voi template va injection | KHONG rieng duoc khoi allele |
| loc pull-up | chay tren profile da hoan chinh | KHONG rieng duoc khoi bat cu lop nao |
| nguong `AT_PC` | ap len tat ca | KHONG rieng duoc |
| **drop-in** | mot nguoi rieng, muc rieng | **hop le** - doc lap that su |

Nen trong bang leave-one-out o Phan 3, chi hai dong doc duoc nhu mot bien doc lap: **su co mat cua allele** va
**drop-in**. Bon dong con lai (n-1, n+1, stutter khac, nhieu nen) deu la mot nua cua mot quan he, va con so lech cua
chung do quan he bi ghep sai, khong phai do luat cua chinh chung sai. Ba "hang muc nho" t liet ke o Phan 3 (nhieu nen
NOC3 +.032, stutter khac NOC3 +.041, n+1 NOC5 -.023) phai doc lai duoi anh sang nay: chung cung la nhung nua-quan-he.

Cach do dung cho cac bien phu thuoc: lay **ca cum** lam don vi. Cum (chieu cao allele + moi stutter sinh tu no) da duoc
do va **khop real** (+.142/.195/.202/.101 so voi +.127/.159/.189/.103, bien do hat giong .016-.029). Cum (nhieu nen +
crowding + shoulder + NOISE_GAIN) chua duoc do nhu mot cum; do la phep do con thieu.

## Phần 4 — chép real BÊN TRONG pipeline, theo tầng, dồn tích

Cách làm (theo yêu cầu: "khi chép real, ta phải tính dựa trên những gì đã chép... chỗ nào cần cái gì thì chép
real vào"): ba điểm móc được đặt bên trong `gen_mixture` của bản sao chẩn đoán `mi_stage.py` —

| điểm móc | vị trí | tầng nào sau nó vẫn do generator sinh |
|---|---|---|
| `alleles` | trước bước artefact | artefact, nhiễu nền, drop-in, lọc, ngưỡng |
| `artefacts` | sau khi artefact chịu survival | nhiễu nền, drop-in, lọc, ngưỡng |
| `noise` | sau khi cộng nhiễu nền | drop-in, lọc pull-up, ngưỡng |

Tầng k nghĩa là: mọi thứ trước k là của real, k và sau k do generator sinh — và **mọi tầng sau đều tính lại từ
cái vừa chép vào**, nên không còn chuyện đổi nhãn bin sau khi sinh (lỗi đã làm vô hiệu mấy vòng trước, vì hai
bên không đồng ý bin nào thuộc lớp nào).

### Lỗi 1 của bộ thử: chèn mà không xoá

Vòng đầu chỉ **gán** giá trị real ở các bin thuộc lớp đó của real, nên đỉnh generator đặt ở những bin real để
trống vẫn còn đứng và cộng thêm: 138.8 rồi 154.4 đỉnh/mẫu so với real 129.5. Sửa: xoá sạch vùng không phải
allele rồi mới gán.

### Lỗi 2 của bộ thử: lọc pull-up bị tính hai lần

Sau khi sửa, **self-check vẫn thất bại**: chép hết trước drop-in/lọc/ngưỡng cho +0.354 +0.449 +0.436 +0.178
(real +0.127 +0.159 +0.189 +0.103) và chỉ 121.6 đỉnh/mẫu so với real 129.5 — mất ~8 đỉnh. Ngưỡng `AT_PC = 3`
chỉ cắt 0.25 đỉnh/mẫu của real (real giữ đỉnh tới p1 = 4 RFU), drop-in chỉ cộng thêm, trần bão hoà không xoá
đỉnh nào. Nên phần mất là của **bộ lọc pull-up cuối**: profile real đã qua bước lọc pull-up của chính phòng
lab, chép nó vào rồi cho generator lọc lần nữa là tính phí lần thứ hai — đúng cái luật "kế thừa rồi chỉ tính
phần tăng thêm". Trong đường sinh bình thường bước này KHÔNG sai (nhiễu được phát ra với P̄ đã chia ra trước),
nó chỉ không được phép chạy trên nội dung đã chép.

**Kết luận về bộ thử:** bộ thử chỉ đọc được khi bước lọc pull-up cuối bị tắt trên nội dung đã chép. Mọi dòng
đo trước khi self-check đạt đều không đọc được.

### Lỗi 3 của bộ thử (nặng nhất): phép chép allele bị ghi đè

Chẩn đoán từng bin ở cấu hình "chép hết, tắt cả ba bước cuối" (120 mẫu, không qua model):

| | |
|---|---|
| thiếu đỉnh (real có, twin không) | **5.78 /mẫu — 100 % ở bin allele** |
| thêm đỉnh (twin có, real không) | 0.00 /mẫu |
| tỉ lệ chiều cao trung vị ở bin chung | 1.000 |

Không đỉnh thừa, chiều cao khớp tuyệt đối ở bin chung, chỉ mất đỉnh ở bin allele — nghĩa là phần allele của
phép chép chưa hề tác dụng. Đọc lại code: điểm móc `alleles` nằm ngay TRƯỚC khối

```python
if _filt:
    contrib = _FH * _sc; contrib_exp = contrib.copy(); contrib_pre = _FHm * _sc
    mix = contrib.sum(0)
```

khối này xây lại `contrib` từ `_FH` của đường filtered, **xoá sạch** mọi thứ vừa chép vào. Hỗn hợp luôn đi
đường filtered, nên mọi dòng "chép allele" của phiên này đều là chép RỖNG; dòng "1 chép allele" (NOC5 −0.117)
thực chất chỉ đo bước ẩn allele ở cuối (`h = 0 ở bin real không có đỉnh`), tức đo SỰ CÓ MẶT, không đo chiều cao.
Sửa: chuyển điểm móc xuống sau khối `_filt` và sau khi nhân `_gain`, nơi không còn bước nào rescale contributions.

### Bộ thử đã sạch — self-check đạt, và các dòng đọc được

Sau ba lần sửa, thêm bước `restore` ở cuối (bin đã chép thì miễn nhiễm với mọi bước sau, vì real chỉ cho TỔNG
mỗi bin), `scratchpad/staged4.py` cho:

| | NOC2 | NOC3 | NOC4 | NOC5 | đỉnh/mẫu |
|---|---|---|---|---|---|
| REAL | +0.127 | +0.159 | +0.189 | +0.103 | 129.5 |
| **TỰ KIỂM: chép hết** | **+0.127** | **+0.159** | **+0.189** | **+0.103** | **129.5** |
| 0 twin, không chép gì | +0.111 | +0.165 | +0.165 | +0.055 | 131.1 |
| 1 chép allele (artefact + nhiễu tính lại) | +0.106 | +0.212 | +0.174 | +0.036 | 130.3 |
| 2 + chép artefact (nhiễu tính lại) | +0.317 | +0.329 | +0.276 | +0.156 | 137.8 |

Hai kết luận:

1. **Tầng allele không phải chỗ chứa khoảng cách.** Chép toàn bộ tầng allele — cả sự có mặt lẫn chiều cao,
   từng người khớp mức của mình trên allele riêng — gần như không đổi gì (NOC5 +0.055 → +0.036, real +0.103;
   NOC3 còn xa hơn). Điều này **bác bỏ** kết luận cũ "twin gap là allele presence": nó đo bằng bộ thử mà phép
   chép allele là chép rỗng, nên cái nó thực sự đo là bước ẩn allele ở cuối.
2. **Phân chia artefact / nhiễu bị sai dù tổng đỉnh đúng.** Chép allele + artefact của real rồi để generator
   sinh nhiễu cho 137.8 đỉnh — thừa 8.3 so với real — và gate vọt lên +0.317/+0.329/+0.276/+0.156. Nhiễu nền
   của generator đang phát ở những bin mà ở real là artefact: tổng khớp vì nhiễu bù cho phần artefact thiếu.
   Đây là đúng loại "pattern sai mà tổng vẫn đúng" khiến model mở thêm một slot decoy.

## Phần 5 — đếm đỉnh theo lớp, real so với twin (chẩn đoán, không fit)

Điểm 2 ở trên đo được trực tiếp mà không cần chép gì. Đếm đỉnh và RFU theo lớp trên 1333 hỗn hợp test, twin
sinh từ chính spec của mỗi mẫu. Lớp định nghĩa theo **panel kiểu gen** của các người đóng góp (không phụ thuộc
allele cha có đứng hay không), nên định nghĩa giống nhau tuyệt đối hai bên — bản đếm theo `classify` thường
(cha phải đứng) cho cùng kết quả, nên lệch không đến từ đổi nhãn.

twin trừ real:

| lớp | NOC2 | NOC3 | NOC4 | NOC5 |
|---|---|---|---|---|
| n−1 số đỉnh | −2.54 | −1.76 | −1.17 | −1.25 |
| n−1 RFU | **−15.4 %** | −8.4 % | −8.9 % | **−17.8 %** |
| nhiễu số đỉnh | +0.76 | +1.56 | **+2.14** | **+2.21** |
| n+1 RFU | −11 % | +6 % | **+71 %** | **+29 %** |
| n−0.5 RFU | ~0 | +33 % | **−58 %** | −25 % |
| allele số đỉnh | −0.12 | +0.77 | +0.59 | +1.69 |

Lệch nhất quán ở MỌI NOC: **thiếu n−1 cả số đỉnh lẫn khối lượng, bù bằng nhiễu nền**. Khối lượng nhiễu thừa
(+18…+41 RFU) không đủ bù phần n−1 thiếu (−128…−278 RFU), nên đây không phải chuyển nhãn mà là thiếu thật.
Hai lệch chỉ có ở NOC4/5: n+1 quá nhiều và quá cao, n−0.5 quá ít.

### Lỗi này CÓ ở NOC1 — nên sửa được từ dữ liệu được phép

Cùng bảng, chạy trên 5992 profile NOC1 của train+val (twin sinh từ chính spec mỗi mẫu; test không đọc):

| lớp | real | twin | lệch đỉnh | RFU real | RFU twin | lệch |
|---|---|---|---|---|---|---|
| **n−1** | 16.62 | 15.04 | **−1.58** | 1669 | 1449 | **−13.2 %** |
| n+1 | 4.41 | 4.66 | +0.25 | 120 | 123 | +2.9 % |
| n−2 | 3.58 | 3.40 | −0.17 | 61 | 51 | −15.5 % |
| n−0.5 | 0.55 | 0.48 | −0.06 | 6 | 4 | −24.5 % |
| n−3 | 2.13 | 2.17 | +0.04 | 23 | 22 | −5.9 % |
| **nhiễu** | 29.18 | 30.30 | **+1.12** | 274 | 290 | +5.6 % |
| allele | 36.11 | 36.36 | +0.25 | 39408 | 39317 | −0.2 % |
| TỔNG | 92.56 | 92.40 | | | | |

Allele khớp, tổng khớp, nhưng n−1 thiếu 1.58 đỉnh và 13.2 % khối lượng, và nhiễu nền phát bù 1.12 đỉnh. RFU
mỗi đỉnh n−1: real 100.4, twin 96.3 — chỉ −4 %, nên phần lớn chỗ hụt là **thiếu đỉnh**, không phải mỗi đỉnh
thấp. n+1 khớp ở NOC1 (lệch NOC4/5 là chuyện khác). Vì lỗi có mặt ở NOC1, nó là lỗi LUẬT và sửa được mà không
chạm vào hỗn hợp nào.

### Định vị: SỰ CÓ MẶT của n−1 ở cha SÁNG, không phải chiều cao

NOC1 train+val, mỗi allele cha đang đứng, phân theo dải chiều cao cha (5992 profile):

| cha (RFU) | số cha/mẫu real→twin | P(n−1 hiện) real | twin | lệch | tỉ lệ n−1/cha real | twin |
|---|---|---|---|---|---|---|
| 0–30 | 1.79 → 2.06 | 0.107 | 0.102 | −0.005 | 0.4333 | 0.4739 |
| 30–60 | 2.55 → 2.49 | 0.137 | 0.119 | −0.018 | 0.2167 | 0.1812 |
| 60–120 | 3.64 → 3.65 | 0.204 | 0.203 | −0.001 | 0.1327 | 0.1164 |
| 120–250 | 4.53 → 4.62 | 0.371 | 0.379 | +0.008 | 0.0909 | 0.0876 |
| 250–500 | 4.42 → 4.43 | 0.629 | **0.570** | −0.059 | 0.0720 | 0.0734 |
| 500–1000 | 4.14 → 4.11 | 0.821 | **0.712** | −0.109 | 0.0634 | 0.0661 |
| 1000–2000 | 3.50 → 3.48 | 0.880 | **0.772** | −0.108 | 0.0606 | 0.0614 |
| >2000 | 4.49 → 4.40 | 0.890 | **0.789** | −0.101 | 0.0607 | 0.0599 |

Tổng do P(hiện): **−1.61 đỉnh/mẫu**, khớp −1.58 của bảng lớp. Tỉ lệ chiều cao khi hiện thì khớp ở mọi dải —
nên luật tỉ lệ ĐÚNG, chỉ sự có mặt sai, và chỉ ở cha từ 250 RFU trở lên. Ở đó n−1 kỳ vọng 30–60 RFU, gấp
10 lần ngưỡng, nên không phải rụng dưới ngưỡng: generator KHÔNG PHÁT.

Nghi phạm: `ART_SURV_P` tra theo chiều cao KỲ VỌNG của artefact — ở 30–60 RFU nó chỉ cho survival 0.65–0.86.
Bảng này fit trên scatter CŨ, còn phiên này đã nới `lsd` và thêm luật stutter theo allele, nên nó đang tính phí
hai lần cùng với ngưỡng (xem `inherit-then-increment`, `step-charge-nested-with-threshold`).

### Nguyên nhân: hệ số tỉ lệ theo allele bị CHẶN, phá chính phép fit của nó

Knockout trên NOC1 loại được cả hai nghi phạm đầu:

| cha (RFU) | P real | twin hiện tại | survival = 1 | lọc pull-up tắt | survival=1 + lọc tắt |
|---|---|---|---|---|---|
| 60–120 | 0.204 | 0.203 | 0.425 | 0.209 | 0.429 |
| 120–250 | 0.371 | 0.379 | 0.653 | 0.378 | 0.654 |
| 250–500 | 0.629 | 0.570 | 0.759 | 0.578 | 0.762 |
| 500–1000 | 0.821 | 0.712 | 0.804 | 0.716 | 0.809 |
| 1000–2000 | 0.880 | 0.772 | 0.818 | 0.780 | 0.816 |
| >2000 | 0.890 | 0.789 | 0.802 | 0.788 | 0.803 |

Lọc pull-up gần như không lấy gì (0.788 so với 0.789). Survival = 1 thì dải dưới 250 vọt quá xa real, nên
survival là CẦN — nhưng ngay cả khi survival = 1 và lọc tắt, cha >1000 RFU vẫn chỉ đạt 0.80–0.82 so với real
0.88–0.89. Đó là TRẦN của tỉ lệ phát, và trần đó không phải giá trị đã hiệu chuẩn:

```python
_rate = np.minimum(rate * ALLELE_RATE[int(d)][sl], 0.99)
```

Trên 203 bin panel có đích n−1: tỉ lệ locus thô trung bình 0.9261, hệ số theo allele trung bình 1.0162, nhân
không chặn 0.9528 — **nhân rồi chặn 0.99 chỉ còn 0.8395**, vì **35 % bin bị chặn**. Mất 0.113, đúng bằng phần
còn lại ở cha sáng. `calibrate_allele_stut.py` fit hệ số nhân có trung bình ≈ 1, nhưng generator chặn ở 0.99
nên phần > 1 mất trắng còn phần < 1 vẫn cắt đủ: phép chặn phá chính phép fit.

Điều này cũng giải thích vì sao CHỈ n−1 bị: nó là offset duy nhất có tỉ lệ gần 1 (0.93). n+1 chỉ 0.077 nên
không bao giờ bị chặn — và đúng là n+1 khớp ở NOC1 (+0.25 đỉnh, +2.9 % RFU).

### Sửa: hệ số theo allele phải sống trên thang odds, không phải hệ số nhân lên tỉ lệ

Thử ba liên kết trên NOC1 (5992 profile):

| cấu hình | thiếu n−1 đỉnh/mẫu |
|---|---|
| hiện tại: `min(rate × f, 0.99)` | **−1.55** |
| `1 − (1−rate)^f` (chặn tự nhiên, hệ số cũ) | −0.62 |
| tắt hẳn hệ số tỉ lệ | **−0.10** |

Nhìn vào `calibrate_allele_stut.py` thì thấy lỗi ở CHÂN ĐẾ, không chỉ ở phép chặn: hệ số được đo bằng *xác suất
có mặt* ở dải cha 100–400 RFU, nơi presence chỉ 0.4–0.6, rồi đem áp làm hệ số nhân lên *tỉ lệ phát* 0.93. Hai
đại lượng khác thang, và thang đích có trần cứng — nên hệ số 1.5 không có chỗ đi, còn hệ số 0.6 vẫn cắt đủ.

Đã sửa (giữ nguyên phần tỉ lệ chiều cao `ALLELE_STUT_OFF`, vốn đã kiểm chéo r = 0.961 / 0.868):
- `calibrate_allele_stut.py` giờ fit **offset logit** của presence, `logit(p_real) − logit(p_twin)`, với
  `var(logit(p̂)) = 1/(n·p·(1−p))` cho phép co Empirical-Bayes; khoá `rate-10`/`rate10` → `lrate-10`/`lrate10`.
- `make_insilico.py` áp nó bằng logit: `rate = σ(logit(rate_bảng) + offset)`. Tự chặn trong (0,1), offset = 0
  cho lại đúng tỉ lệ của bảng, và cả hai phía cùng một đại lượng (odds của presence).

### Sửa lần hai: fit ở dải mà generator vốn đã khớp

`_CARR_MULT` (hệ số theo số donor panel mang bin) được chuẩn hoá về trung bình 1 trên tập mà `ART_SURV_P` được
fit — bảng đó chuẩn hoá cho đỉnh bằng 1 nên trần phải do `rate` cung cấp. Hoá ra trung bình có trọng số vốn đã
≈ 1, nên đây là no-op; giữ lại chỉ để chân đế tường minh. Và tỉ lệ locus có trọng số ở cha sáng là 0.8947 so
với real 0.8850 — **bảng tỉ lệ vốn đã đúng**, mọi hệ số < 1 nhân thêm đều làm twin thiếu.

Nguyên nhân thật của phần dư: **dải fit**. Offset đo ở cha 100–400 RFU, đúng dải mà generator đang cao hơn real
(120–250: 0.415 vs 0.371), nên nó hút lấy lỗi hình dạng rồi mang sang đầu sáng, nơi generator vốn đã hơi thiếu.
Đổi sang dải 250–500, nơi generator vốn khớp (0.636 vs 0.629) — theo luật `calibrate-single-process-positions`:

| bước | thiếu n−1 đỉnh/mẫu |
|---|---|
| ban đầu (nhân + chặn 0.99) | −1.55 |
| offset logit, fit ở 100–400 | −0.63 |
| **offset logit, fit ở 250–500** | **−0.38** |
| (đối chứng: tắt hẳn luật) | −0.10 |

Dải fit giờ khớp chính xác (0.628 vs 0.629). Phần dư −0.38 là **lỗi hình dạng của `ART_SURV_P`**: vẫn quá rộng
tay ở cha 120–250 (+0.041) và quá dè ở cha > 500 (−0.036…−0.047). Đó là bước kế tiếp, không phải luật theo allele.

### Kết quả của phần sửa, và cái nó làm lộ ra

Bảng lớp trên NOC1 sau khi sửa:

| lớp | real | twin trước | twin sau | RFU real | RFU trước | RFU sau |
|---|---|---|---|---|---|---|
| **n−1** | 16.62 | 15.04 (−1.58) | **16.27 (−0.35)** | 1669 | 1449 (−13.2 %) | **1619 (−3.0 %)** |
| nhiễu | 29.18 | 30.30 (+1.12) | 30.11 (+0.94) | 274 | 290 | 289 |
| n−2 | 3.58 | 3.40 | 3.43 | 61 | 51 | 52 |
| TỔNG | 92.56 | 92.40 | 93.41 | | | |

Gate trên 1333 hỗn hợp test:

| | NOC2 | NOC3 | NOC4 | NOC5 | đỉnh/mẫu |
|---|---|---|---|---|---|
| REAL | +0.127 | +0.159 | +0.189 | +0.103 | 129.5 |
| twin trước | +0.111 | +0.165 | +0.165 | +0.055 | 131.1 |
| twin sau | +0.158 | +0.192 | +0.128 | +0.035 | **133.1** |

Gate XA real hơn. Đúng như bảng NOC1 báo trước: **nhiễu nền vốn phát thừa để bù cho n−1 thiếu**, nên trước đây
tổng trông đúng nhờ hai lỗi triệt tiêu nhau. Sửa một bên thì lỗi bên kia lộ ra — đây là kết quả ĐÚNG về mặt
từng-con-số (n−1 khớp real trên NOC1) và nó chỉ đúng vào mục tiêu tiếp theo.

Đã loại: cặp pull-up (phóng tỉ lệ nhiễu lên 1/(1−P̄) = 1.0963 rồi để bộ lọc lấy lại). Tắt cả hai phía chỉ đổi
nhiễu 30.11 → 30.06 và tổng 93.41 → 93.21, nên cặp đó cân. Phần thừa 0.9 đỉnh/mẫu là của chính tỉ lệ nhiễu nền.

### Bảng hỗn hợp sau khi sửa, và thứ tự ưu tiên còn lại

twin trừ real, số đỉnh/mẫu (lớp theo panel):

| lớp | NOC2 | NOC3 | NOC4 | NOC5 |
|---|---|---|---|---|
| n−1 (trước) | −2.54 | −1.76 | −1.17 | −1.25 |
| **n−1 (sau)** | **−0.49** | **−0.08** | **+0.90** | **+0.25** |
| n−1 RFU (sau) | +0.3 % | +3 % | +4 % | −10 % |
| **nhiễu** | **+1.01** | **+2.20** | **+1.49** | **+1.92** |
| **allele** | +0.07 | **+1.17** | **+1.38** | **+1.89** |
| n+1 | −0.15 | +0.40 | **+1.04** (RFU +77 %) | +0.37 |
| n−2 | +0.01 | +0.19 | +0.26 | +0.45 |
| n−0.5 | −0.02 | −0.03 | **−0.26** (RFU −63 %) | −0.14 |
| n−3 | +0.10 | +0.17 | +0.09 | +0.31 |
| TỔNG | +0.53 | +4.01 | +4.89 | +5.04 |

Thứ tự ưu tiên tiếp theo, theo độ lớn đo được:
1. **nhiễu nền phát thừa** 1.0–2.2 đỉnh/mẫu ở mọi NOC (và +0.9 ở NOC1) — lớn nhất, và không phải do cặp pull-up
2. **twin giữ nhiều allele hơn real** +1.2/+1.4/+1.9 ở NOC3/4/5
3. **n+1 quá nhiều và quá cao, riêng NOC4** (+1.04 đỉnh, RFU +77 %)
4. **n−0.5 quá ít ở NOC4/5** (RFU −63 % ở NOC4)
5. hình dạng `ART_SURV_P` theo chiều cao (phần dư −0.38 của n−1 trên NOC1)

## Phần 6 — nhiễu nền: pipeline và leave-one-out chép đáp án

### Pipeline sinh một đỉnh nhiễu

Sự có mặt (`_pb`, rồi một phép Bernoulli mỗi bin):

| # | bước | đầu vào |
|---|---|---|
| P1 | `_NL_P`: tỉ lệ nền theo locus và theo bin, bin phân số thấp hơn | hiệu chuẩn NOC1 |
| P2 | tăng theo vai: `× interp(log(_shN), log(SHOULDER_H), SHOULDER_NOISE)` | **chiều cao allele đang đứng** |
| P3 | `× _CR_MAX` (phát dư, chờ H3 tỉa lại) | hằng 1.4941 |
| P4 | `÷ (1 − _PBAR_NOISE)` bù trước cho bộ lọc pull-up | hằng, trung bình 1.0963 |
| P5 | `= 0` ở mọi bin người đóng góp sở hữu | panel |
| P6 | phép rút Bernoulli | |

Chiều cao:

| # | bước | đầu vào |
|---|---|---|
| H1 | thang phân vị theo locus `NOISE_LOC_Q`; một offset chung cả profile, ghép với offset artefact qua `RUN_COUPLE`; tương quan từng đỉnh `NOISE_RHO` | |
| H2 | `NOISE_GAIN` × dư lượng log(tổng quan sát − template − injection) | tổng RFU, ng, injection |
| H3 | tỉa theo đám đông `interp(_nh, NOISE_CR_H, _col) / _CR_MAX`, `_col` tra theo **độ chiếm chỗ `(mix ≥ AT_PC).sum()`**; `NOISE_CR_EXT` ngoại suy khi vượt dải | **cả profile đã lắp** |

Ở tầng 2 của bộ thử (allele và artefact chép từ real), P2 và H3 đọc chính profile của real — đáp án đã chép sẵn.

### Bảng leave-one-out (tự kiểm đạt tuyệt đối)

| dòng | gate NOC2/3/4/5 | đỉnh nhiễu/mẫu | RFU nhiễu/mẫu |
|---|---|---|---|
| REAL | +0.127 +0.159 +0.189 +0.103 | 22.64 17.94 15.62 15.12 | 208 158 148 157 |
| **tự kiểm: chép cả nhiễu** | +0.127 +0.159 +0.189 +0.103 | 22.64 17.94 15.62 15.12 | 208 158 148 157 |
| nền: chỉ nhiễu do generator | +0.149 +0.197 +0.192 +0.098 | 22.56 **19.39 17.34 16.40** | 236 209 175 170 |
| P2 tắt | +0.168 +0.191 +0.210 +0.103 | 26.03 23.19 20.98 20.43 | 265 233 213 208 |
| P3+H3 tắt | +0.145 +0.170 +0.195 +0.100 | 22.54 18.95 17.31 16.36 | 226 192 177 162 |
| **P4 tắt** | +0.146 +0.178 +0.188 +0.104 | 21.19 **17.89 15.97 15.29** | 219 184 166 150 |
| H2 tắt | +0.166 +0.192 +0.204 +0.112 | 22.84 19.60 17.69 17.08 | 229 199 182 176 |
| H3b tắt (ngoại suy) | +0.156 +0.195 +0.183 +0.099 | 22.80 19.52 17.31 16.07 | 235 204 171 156 |

1. **P4 là phần thừa phẳng +9.6 %**: tắt nó đưa NOC3/4/5 về đúng real nhưng làm NOC2 thiếu 1.45 đỉnh — nên nó
   là một phần của lỗi, không phải toàn bộ.
2. **Cặp đám đông P3+H3 gần như vô tác dụng**: tắt cả hai chỉ đổi 22.56 → 22.54 và 19.39 → 18.95. Nó phóng tỉ lệ
   lên `_CR_MAX` = 1.4941 rồi tỉa lại đúng chừng đó, **không sinh ra phản ứng theo độ chiếm chỗ** — mà đó là lý do
   duy nhất nó tồn tại. Real giảm 22.64 → 15.12 theo NOC, generator chỉ giảm 22.56 → 16.40.

### Chỗ ngồi là biến đại diện, không phải biến nhân quả

Đo đường phản ứng ở tầng 2 (chỉ nhiễu do generator sinh, chỗ ngồi và vai đều đọc profile của real):

| chỗ ngồi | n mẫu | tỉ lệ/bin real | twin | twin/real |
|---|---|---|---|---|
| 60–90 | 35 | 0.0722 | 0.0716 | 0.991 |
| 90–120 | 353 | 0.0619 | 0.0630 | 1.017 |
| 120–150 | 758 | 0.0501 | 0.0540 | 1.079 |
| 150–180 | 186 | 0.0402 | 0.0449 | 1.117 |

Generator đúng ở chỗ ngồi thấp, thừa 8–12 % ở chỗ ngồi cao. Lý do cơ học: `NOISE_CR_S` chỉ tới **95** còn hỗn hợp
ngồi 120–180, nên `np.interp` **kẹp** ở cột cuối, và `NOISE_CR_EXT = [0,0,0,0,0]` nên **không có ngoại suy nào**.
Bản thân bảng cũng gần phẳng theo chỗ ngồi (h≈4: 0.934 → 0.955; h≈10: 1.187 → 1.134).

Nhưng phân tầng trên real cho thấy **chỗ ngồi không phải nguyên nhân**:

| chỗ ngồi | NOC1 | NOC2 | NOC3 | NOC4 | NOC5 |
|---|---|---|---|---|---|
| 100–115 | 0.0641 | 0.0630 | 0.0608 | 0.0618 | 0.0619 |
| 115–130 | 0.0642 | 0.0552 | 0.0524 | 0.0567 | 0.0611 |
| 130–145 | **0.0661** | 0.0587 | 0.0482 | 0.0461 | 0.0475 |
| ≥145 | **0.0796** | | 0.0425 | 0.0405 | 0.0397 |

Ở cùng chỗ ngồi, NOC1 **tăng** (0.0641 → 0.0796) còn hỗn hợp **giảm** (→ 0.0397); còn trong nội bộ hỗn hợp, ở
cùng chỗ ngồi thì các NOC bằng nhau. Một NOC1 ngồi 145 ngồi đông vì nhiều đỉnh mờ; một hỗn hợp ngồi 145 ngồi
đông vì nhiều allele thật, với template gấp 2.5–6 lần (ng trung vị 0.062 so với 0.15–0.37). Nên biến nhân quả
phải là lượng tín hiệu/template, không phải số đỉnh — xem `stratify-by-causal-variable`.

### Biến nhân quả: số đỉnh ≥ 50 RFU, và lệch chân đế ở trục tra bảng

Thử 10 biến, mỗi biến chia 6 thập phân vị trên toàn bộ mẫu rồi so tỉ lệ NOC1 với tỉ lệ hỗn hợp **trong cùng ô**
(biến đúng là biến có hai hàng khớp nhau):

| biến | trung bình \|lệch NOC1 − hỗn hợp\| |
|---|---|
| **số đỉnh ≥ 50 RFU** | **0.0035** |
| số đỉnh ≥ 100 RFU | 0.0042 |
| trung vị chiều cao | 0.0052 |
| chỗ ngồi (≥ AT) | 0.0067 |
| số allele đang đứng | 0.0076 |
| log tổng RFU | 0.0098 |
| dư lượng NOISE_GAIN | 0.0175 |

Đường gộp theo số đỉnh ≥ 50 RFU (NOC1 fit được, hỗn hợp chỉ để chẩn đoán):

| đỉnh ≥ 50 | n NOC1 | n hỗn hợp | tỉ lệ NOC1 | tỉ lệ hỗn hợp |
|---|---|---|---|---|
| 0–20 | 1420 | 30 | 0.0794 | 0.0796 |
| 20–30 | 986 | 37 | 0.0768 | 0.0750 |
| 30–40 | 1176 | 47 | 0.0721 | 0.0740 |
| 40–50 | 1374 | 67 | 0.0656 | 0.0695 |
| 50–60 | 825 | 96 | 0.0624 | 0.0651 |
| 60–70 | 889 | 133 | 0.0601 | 0.0592 |
| 70–80 | 657 | 168 | 0.0574 | 0.0566 |
| 80–90 | 143 | 170 | 0.0601 | 0.0522 |
| 90–100 | 23 | 174 | — | 0.0473 |
| ≥100 | 0 | 411 | — | 0.0383 |

Và lỗi gốc là **lệch chân đế ở trục tra bảng**: `calibrate.py` fit đường theo
`(_on & (_own | _prd)).sum()` — số đỉnh đứng ở **bin tín hiệu** — còn generator tra bằng
`(mix >= AT_PC).sum()` — **toàn bộ số đỉnh**. Tổng luôn lớn hơn, nên mọi hỗn hợp rơi quá cột cuối (95) và bị kẹp,
mà `noise_cr_inc.json` lại ghi `NOISE_CR_EXT = 0` nên ngoài dải không còn đàn áp nào. Tệ hơn, tổng số đỉnh **tự
quy chiếu** (đếm cả nhiễu do chính bước này sinh), nên trên NOC1 nó chạy ngược chiều.

Đã sửa: trục của cả `calibrate_crowd.py` và `make_insilico.py` đổi sang **số đỉnh ≥ `NOISE_CR_TH` = 50 RFU**,
`calibrate_crowd.py` ghi đúng điểm giữa từng cột và ghi **độ dốc ngoại suy log-tuyến tính** (trước bị ghi bằng 0).

### Kết quả phần nhiễu nền

`calibrate_crowd.py` chạy lại trên trục mới cho bảng tăng thêm **đơn điệu** ở dải mờ nhất (h 0–8):
0.999 / 0.961 / 0.957 / 0.926 / 0.910 / 0.867 qua các cột 13.1 / 29.6 / 39.9 / 48.9 / 62.4 / 75.8, và độ dốc
ngoại suy −0.00207 mỗi đỉnh cao (trước ghi bằng 0). Trên trục CŨ bảng gần như phẳng (×1.04 thưa, ×0.99 đông) —
điều mà chú thích trong file từng coi là "không có gì để kéo dài"; đó là hệ quả của trục sai, không phải của dữ liệu.

`rng.random()` trong phần ngoại suy được giữ nguyên: ngoài dải NOC1 không có gì đo được, nên mỗi profile rút một
điểm **giữa** "giữ ở biên" và "kéo dài xu thế" — đúng cách tiếp cận theo khoảng.

Nhiễu thừa (twin − real, đỉnh/mẫu) và tổng đỉnh thừa:

| | NOC2 | NOC3 | NOC4 | NOC5 |
|---|---|---|---|---|
| nhiễu, trước | +1.01 | +2.20 | +1.49 | +1.92 |
| **nhiễu, sau** | **+0.25** | **+1.10** | +1.64 | **+1.23** |
| tổng, trước | +0.53 | +4.01 | +4.89 | +5.04 |
| **tổng, sau** | **−0.90** | **+2.74** | **+3.39** | **+4.12** |

NOC1: nhiễu 30.11 → 29.99 (real 29.18), tổng 93.41 → 93.17.

Gate trên 1333 hỗn hợp:

| | NOC2 | NOC3 | NOC4 | NOC5 | đỉnh/mẫu |
|---|---|---|---|---|---|
| REAL | +0.127 | +0.159 | +0.189 | +0.103 | 129.5 |
| twin ban đầu (hai lỗi triệt tiêu nhau) | +0.111 | +0.165 | +0.165 | +0.055 | 131.1 |
| sau khi sửa n−1 | +0.158 | +0.192 | +0.128 | +0.035 | 133.1 |
| **sau khi sửa cả nhiễu** | **+0.118** | **+0.140** | **+0.188** | +0.028 | **131.8** |

Phần dư của nhiễu là **giới hạn ngoại suy thật**: real giảm −0.0052/đỉnh trong dải NOC1 nhưng −0.0101/đỉnh ở
vùng chỉ hỗn hợp với tới. Không được fit chỗ đó bằng hỗn hợp; luật giờ đúng chân đế và đi đúng chiều, thế là đủ.

### Ưu tiên còn lại (twin − real trên hỗn hợp)

1. **n+1 ở NOC4**: +1.06 đỉnh và RFU **+82 %** (118 so với 65) — lỗi lớp lớn nhất còn lại; ở NOC1 thì n+1 khớp
2. **allele giữ quá nhiều ở NOC5**: +1.50 đỉnh (98.71 vs 97.21), NOC3 +0.87, NOC4 +0.37
3. **nhiễu dư** +1.10…+1.64 ở NOC3/NOC4
4. **n−0.5 thiếu** ở NOC4 (RFU 9 so với 19) và **n−2 thiếu RFU** (−15 % cả trên NOC1)
5. **n−1 RFU thiếu 8.5 % riêng ở NOC5** (1034 so với 1130) dù số đỉnh khớp
6. hình dạng `ART_SURV_P` (phần dư −0.38 đỉnh n−1 trên NOC1)

### Không có luật riêng của hỗn hợp — kiểm trực tiếp

Một hỗn hợp là **một ống, một lần chạy**, chỉ gồm nhiều nguồn cộng lại; nên nếu nhiễu nền cần độ dốc dốc hơn ở
vùng chỉ hỗn hợp với tới, đó phải là NOC1 bị đo sai ở đầu sáng, không phải vật lý mới.

**Pull-up không phải nguyên nhân.** `calibrate.py` từng nêu nghi vấn ("a rise is pull-up riding on brightness").
Loại các bin ứng viên pull-up (bin có láng giềng khác màu trong 2 bp mang đỉnh cao) ở CẢ HAI phía làm đường NOC1
**phẳng hơn** chứ không dốc thêm: độ dốc −0.00526 → −0.00470 → −0.00433 → −0.00398 khi hạ ngưỡng loại từ 2000 →
1000 → 500 RFU, và tỉ số hỗn hợp/NOC1 ở cột 80–90 **xấu đi** 0.870 → 0.790 → 0.801 → 0.810.

**Hai lực ngược nhau, nên không biến đơn nào gộp được.** Một NOC1 có 85 đỉnh cao chỉ có 36 allele — 49 đỉnh cao
còn lại là stutter của những allele cực sáng, tức một ống rất đậm; một hỗn hợp 85 đỉnh cao có 60–97 allele ở mức
trung bình. Độ sáng của lần chạy làm nền cao hơn nên NHIỀU đỉnh nhiễu vượt ngưỡng; tải tín hiệu làm phần mềm gọi
đỉnh khắt khe hơn nên ÍT hơn. Khớp cả hai biến:

| đỉnh ≥ 50 | h 0–9 | h 9–11 | h 11–13 |
|---|---|---|---|
| 0–40 | 0.0772/0.0759 → **0.98** | 0.0754/0.0759 → **1.01** | — |
| 40–60 | 0.0641/0.0670 → **1.05** | 0.0649/0.0677 → **1.04** | — |
| 60–80 | 0.0593/0.0576 → **0.97** | 0.0590/0.0583 → **0.99** | 0.0585/0.0582 → **0.99** |
| ≥80 | — | 0.0601/0.0444 → 0.74 | 0.0592/0.0450 → 0.76 |

(mỗi ô: tỉ lệ NOC1 / tỉ lệ hỗn hợp → tỉ số)

Ở mọi ô có đủ dữ liệu với dưới 80 đỉnh cao, hai bên **bằng nhau trong 3–5 %**. Lệch chỉ còn ở ô ≥ 80 đỉnh cao,
nơi NOC1 có 166 trong 7493 mẫu và toàn là ống cực đậm. Vậy **không có luật riêng của hỗn hợp**; phần dư là vấn
đề DẢI DỮ LIỆU, và đúng chỗ mà code đã mô hình hoá bằng một khoảng (rút đều giữa "giữ ở biên" và "kéo dài xu thế").

### Khoảng ngoại suy bao được NOC4/NOC5, KHÔNG bao được NOC3

Đo hai đầu của khoảng (số đỉnh nhiễu/mẫu ở tầng 2):

| | NOC2 | NOC3 | NOC4 | NOC5 |
|---|---|---|---|---|
| REAL | 22.64 | 17.94 | 15.62 | 15.12 |
| giữ ở biên (ngoại suy = 0) | 22.65 | 19.30 | 16.98 | 16.26 |
| kéo dài hết (ngoại suy = 1) | 22.48 | **19.26** | 16.17 | 15.24 |

NOC4 và NOC5 nằm trong khoảng; NOC3 thì **cả hai đầu đều thừa +1.32** vì NOC3 ít đỉnh cao nên phần ngoại suy
gần như không tác động. Theo đúng nguyên tắc "không có luật riêng của hỗn hợp", phần thừa ở NOC3 phải nằm trong
dải NOC1 đo được — và đúng thế: NOC1 cũng thừa +0.82 đỉnh (+2.8 %).

Nghi phạm là **P4**: nó phát thừa 9.6 % (`1/(1 − _PBAR_NOISE)`, `_PBAR_NOISE` trung bình 0.0805) để chờ bộ lọc
pull-up lấy lại, nhưng đo trực tiếp thì bộ lọc lấy ~3 đỉnh/mẫu (~14 %). Hai con số không khớp. Tắt P4 đưa NOC3
xuống −0.38, để nguyên là +1.29, nên hệ số đúng nằm quanh 1.02. Phải đo lại như một increment trên NOC1 (đếm
đúng số đỉnh nhiễu bộ lọc bỏ, so với số P4 cộng vào, theo từng điều kiện) chứ không chọn bằng hỗn hợp.

### Sổ kế toán chính xác của nhiễu nền (một lượt chạy, không ghép knockout)

Tắt P4 làm số lần rút thay đổi nên hai lượt knockout không ghép cặp được (`rng-knockout-desync`). Thay vào đó ghi
số trực tiếp trong chính lượt chạy. Tầng 2 cho hỗn hợp (allele + artefact chép từ real), NOC1 chạy bằng spec riêng:

| NOC | bin trống | E trước P4 | E sau P4 | E ở bin trống | rút | qua tỉa | lọc bỏ | thấy | REAL | thừa |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 423 | 74.83 | 81.91 | 75.43 | 75.51 | 43.94 | 3.41 | 30.08 | 29.43 | +0.65 |
| 2 | 374 | 71.49 | 78.34 | 68.07 | 68.38 | 38.53 | 5.06 | 22.38 | 22.64 | −0.26 |
| 3 | 348 | 69.64 | 76.37 | 63.11 | 63.45 | 35.23 | 5.39 | 19.23 | 17.94 | +1.29 |
| 4 | 325 | 67.93 | 74.53 | 59.62 | 60.26 | 33.12 | 5.89 | 16.85 | 15.62 | +1.23 |
| 5 | 311 | 68.32 | 74.94 | 58.39 | 58.18 | 32.50 | 5.63 | 16.27 | 15.12 | +1.14 |

**Giả thuyết P4 SAI, và sai ngược chiều.** Tỉ lệ bộ lọc bỏ tăng theo NOC — 7.8 % (NOC1), 13.1 / 15.3 / 17.8 /
17.3 % (NOC2–5) — còn P4 dùng hằng 1.095, tức giả định bỏ 8.7 %. Ở NOC1 nó gần đúng (cần 1.084); ở NOC3–5 nó
**bù thiếu** (cần 1.18–1.22). Sửa P4 cho đúng sẽ làm phần thừa TO THÊM. Bỏ giả thuyết này.

**Chỗ thiếu là bước tỉa.** Tỉ lệ giữ chỉ đi 0.582 (NOC1) → 0.559 (NOC5), tức 4 %. Tính trên mỗi bin trống, twin
giảm 0.0711 → 0.0523 (−26 %) còn real giảm 0.0696 → 0.0486 (−30 %): dư +2.2 % ở NOC1 lên +7.6 % ở NOC5, và nó
nằm ở dải 76–99 đỉnh cao. Ở đó NOC1 CÓ dữ liệu (657 profile ở 70–80, 166 trên 80) nhưng cột cuối của
`calibrate_crowd` gộp mọi thứ ≥ 70 thành một điểm, làm đường phẳng **do cách chia cột**, không do dữ liệu.

Chia cột trên thành 70–82 và 82+: dải mờ nhất tiếp tục giảm **0.910 → 0.891 → 0.720** (cột cuối n = 79), cột
trung bình 74.0 và 86.6, độ dốc ngoại suy −0.00209. Trước khi chia, đường đứng ở 0.867 từ 75.8 trở đi.

### Chốt phần nhiễu: chia cột + chuẩn hoá theo cấu trúc

Sau khi chia cột, `_CR_MAX = NOISE_CR_M.max()` bị **ô sáng chiếm quyền**: dải h ≥ 25 của cột 82+ đọc 4.08 (212
đỉnh real so với 52 của twin). Giá trị đó là thật nhưng **không phải luật đám đông** — cả hàng đó chạy 0.16 /
0.25 / 0.27 / 0.32 / 0.46 / 1.13 / 4.08, tức real có ÍT đỉnh nhiễu cao hơn twin ở profile yên và NHIỀU hơn nhiều
ở profile sáng: đó là độ sáng. Để nó đặt hằng chuẩn hoá thì mọi phép phát bị phóng 4.4× thay vì 1.5× và hình
dạng nhiễu sống sót đổi hẳn.

`_CR_MAX` chỉ là hằng chuẩn hoá để `_keep ≤ 1`, và `_keep` giờ được clip ở 1, nên nó lấy từ **hai dải mờ nhất**
— những dải mang số đếm và cư xử như luật đám đông (0.70–1.28) — cho `_CR_MAX = 1.279`. Chọn theo cấu trúc,
không theo bất kỳ điểm số nào trên hỗn hợp.

| cấu hình | NOC1 | thừa nhiễu NOC2/3/4/5 | tr.bình |
|---|---|---|---|
| chỉ sửa trục | +0.65 | −0.26/+1.29/+1.23/+1.14 | 0.98 |
| + chia cột, `_CR_MAX` = 4.08 | +0.44 | +0.56/+1.34/+1.07/+0.61 | 0.90 |
| + chia cột, cap 1.6 | +0.81 | −0.43/+0.97/+0.30/+0.79 | 0.62 |
| **+ chia cột, `_CR_MAX` từ dải mờ = 1.279** | **+0.55** | **+0.32/+0.88/+0.78/+0.27** | **0.56** |

Bản `_CR_MAX` = 4.08 nhích hơn trên NOC1 (+0.44) nhưng sai cấu trúc: 79 mẫu đặt hằng chuẩn hoá rồi phát dư 4.4×
mà `_keep` vẫn bị clip, tức chỉ thêm phương sai. Giữ lựa chọn theo cấu trúc.

Từ đầu phần nhiễu: thừa nhiễu NOC2–5 **+1.01/+2.20/+1.49/+1.92 → +0.32/+0.88/+0.78/+0.27**, NOC1 +0.94 → +0.55.

### Trạng thái sau khi sửa nhiễu nền

NOC1: n−1 −0.43 đỉnh (RFU −3.4 %), nhiễu +0.77 (+4.9 %), n−2 RFU −14 %, n−0.5 RFU −24 %, tổng +0.60.

Hỗn hợp, twin − real (đỉnh/mẫu):

| lớp | NOC2 | NOC3 | NOC4 | NOC5 |
|---|---|---|---|---|
| **allele** | −0.24 | **+1.17** | +0.52 | **+1.76** |
| nhiễu | +0.19 | +0.49 | +0.79 | +1.31 |
| **n+1** | −0.05 | +0.24 | **+1.02** (RFU **+72 %**) | +0.44 |
| n−1 | −0.35 | −0.06 | +0.35 | +0.16 |
| n−1 RFU | −1 % | +5.3 % | +3.2 % | **−8.0 %** |
| n−0.5 RFU | 0 % | +33 % | **−58 %** | −17 % |
| n−3 | +0.07 | +0.19 | +0.16 | +0.34 |
| TỔNG | −0.34 | +2.19 | +2.77 | +4.18 |

Gate: +0.159 / +0.168 / +0.162 / +0.045 so với real +0.127 / +0.159 / +0.189 / +0.103.

### Ưu tiên tiếp theo, đọc qua nguyên tắc "không có luật riêng của hỗn hợp"

1. **allele giữ quá nhiều, +1.76 ở NOC5 và +1.17 ở NOC3** — lớp sai nhiều nhất. NOC1 chỉ +0.19, nên đây là một
   lượng theo-ống mà generator đặt sai (mức của người mờ nhất), không phải luật mixture.
2. **n+1 ở NOC4: RFU +72 %** (112 so với 65) trong khi NOC1 khớp (+2.9 %). n+1 được đúc từ chiều cao allele, nên
   theo nguyên tắc trên, hoặc phân bố chiều cao allele ở NOC4 sai, hoặc bộ donor của NOC4 khác — kiểm theo bộ donor
   (`noc-levels-few-donor-sets`) trước khi đổi luật nào.
3. **n−0.5**: RFU −24 % ngay trên NOC1, −58 % ở NOC4. Sửa được từ NOC1.
4. **n−2**: RFU −14 % trên NOC1. Sửa được từ NOC1.
5. nhiễu còn dư +0.5…+1.3, phần lớn ở dải ngoài tầm NOC1 (đã là khoảng, không fit thêm).
6. **n−1 RFU: +5.3 % ở NOC3 nhưng −8.0 % ở NOC5** — đổi dấu, nên là hiệu ứng theo bộ donor chứ không phải luật.

## Phần 7 — bảng hai cấp: giới hạn lỗi về tầng, rồi về hàm

### Pipeline hoàn chỉnh, ba tầng

**Tầng ALLELE** — `_filter_contrib` cho từng người, rồi các bước cấp ống:

| # | bước | nội dung |
|---|---|---|
| A1 | `_s_tube` | một lần rút hư hại cho cả ống (`FILT_S`) |
| A2 | phần tế bào | `_nc ~ Poisson(R·φ·ng/FILT_PG)`, `R ~ logU(CELL_R)` → φ thành phần tế bào thực |
| A3 | `_eff` | `κ ~ U(0, KAPPA_MAX)`, `_eff = 2^(−κ(k−1))`, một hiệu suất cho cả ống |
| A4 | `_filter_source` | chọn một lần chạy NOC1 thật của người đó (mang sẵn degradation và injection riêng) |
| A5 | `lam_s` | số bản kỳ vọng ở nguồn: `t_s·q_s^(−a·x)/pg·dose`, `x = (size−80)/134` |
| A6 | `per` | RFU trên mỗi bản = chiều cao quan sát của nguồn / `lam_s` |
| A7 | `a_t` | số mũ hư hại của ống: `FILT_A[họ]·2^(α + β·log q_tube + s_tube + N(0,sd_d))` |
| A8 | `lam_t` | số bản kỳ vọng ở đích: `eff·t·q_tube^(−a_t·x)/pg·dose` |
| A9 | `k ~ Poisson(lam_t)` | **xổ số sự có mặt** từng allele |
| A10 | `HEIGHT_SHARE` | số bản dùng chung theo locus `_sh ~ Poisson(mean lam_t)`, trộn vào `k_h` |
| A11 | `ksum ~ Gamma(k_h/CV², CV²)` | tán xạ khuếch đại |
| A12 | `_INJ_STEP` | hệ số bước injection `f` |
| A13 | `_jit`, `sd_al` | jitter một lần cho cả run, cộng jitter từng allele |
| A14 | `H = where(k ≥ 1, per·ksum·f·jit·jitter, 0)` | chiều cao, bằng 0 nơi không có bản nào |
| A15 | `_bud`, `_sc` | giải ngân sách để tổng khớp `t_total` |
| A16 | `_pb_sum` | chuẩn hoá tổng phần về 1 |
| A17 | `_gain` | hệ số injection, đưa về thang capillary tuyệt đối |

**Tầng ARTEFACT** (stutter mọi loại):

| # | bước | nội dung |
|---|---|---|
| S1 | `art_table_loc` | theo locus: rate, `lmu`, `lsd`, độ dốc theo số allele |
| S2 | `ALLELE_STUT_OFF` | offset tỉ lệ theo từng allele |
| S3 | `ALLELE_RATE` | offset **logit** trên tỉ lệ phát, theo từng allele |
| S4 | `STUT_LT_A/G/CAP` | nâng tỉ lệ ở cha mờ |
| S5 | `_roff · _rsd` | một offset chung cho cả run |
| S6 | `_loff`, `ART_LOCUS_R` | tầng locus |
| S7 | `lsd · ART_SCAT_K · ART_SCAT_D` | tán xạ từng lần rút |
| S8 | `STUT_LT_H0` | nhân gamma theo cha mờ |
| S9 | `ART_CAP` | chặn trên từng lần rút |
| S10 | `_cross(d)` | hệ số chéo |
| S11 | survival | `ART_SURV_P(art_h)` × `_CARR_MULT` × `SHOULDER_ART` × `STUT_SMALL_Q` |
| S12 | cộng vào `mix` | một quyết định survival mỗi bin, trên chiều cao đã thực hiện |

**Tầng NHIỄU NỀN**: P1–P6 và H1–H3, đã liệt kê ở Phần 6.

### Cấp 1 — chép hết, sinh đúng MỘT tầng

| dòng | allele | artefact | nhiễu |
|---|---|---|---|
| tự kiểm | chép | chép | chép |
| L1-A | **sinh** | chép | chép |
| L1-S | chép | **sinh** | chép |
| L1-N | chép | chép | **sinh** |

### Cấp 1 — kết quả

Một lỗi trong bộ thử phải sửa trước: dòng `h = np.where(own & (z <= 0), 0.0, h)` ẩn những bin allele real không
có đỉnh. Ở L1-A thì tầng allele là tầng ĐANG ĐƯỢC ĐO, nên ẩn chúng là chép đáp án về đúng cái cần đo (tàn dư từ
bộ thử cũ, nơi allele được chép). Bỏ đi thì số đỉnh allele đổi dấu: 90.7 → 99.0 ở NOC5.

| dòng | gate NOC2/3/4/5 | lệch gate | đỉnh của tầng được sinh | RFU tầng đó |
|---|---|---|---|---|
| REAL | +0.127 +0.159 +0.189 +0.103 | — | 61.7 / 80.4 / 91.7 / 97.2 (allele) | — |
| tự kiểm (chép cả 3) | +0.127 +0.159 +0.189 +0.103 | 0 | đúng | đúng |
| **L1-A: sinh ALLELE** | **+0.383 +0.505 +0.496 +0.364** | **+0.26 +0.35 +0.31 +0.26** | 61.5 / 81.6 / 92.2 / 99.0 | −0.7…−1.0 % |
| L1-S: sinh ARTEFACT | +0.129 +0.223 +0.200 +0.050 | +0.00 +0.06 +0.01 −0.05 | 31.13 / 31.04 / 32.79 / 27.04 | −1.0…+5.9 % |
| L1-N: sinh NHIỄU | +0.157 +0.181 +0.185 +0.101 | +0.03 +0.02 −0.00 −0.00 | 22.79 / 18.60 / 16.76 / 15.31 | +0.3…+21 % |

**Tầng allele mang gần như toàn bộ lỗi gate**, lớn hơn stutter và nhiễu một bậc. Và nó KHÔNG phải lỗi số lượng
hay tổng khối lượng: số đỉnh allele lệch dưới 2 và RFU lệch dưới 1 %. Lỗi ở **cách phân bố**.

### Cấp 2 — tách tầng allele thành ba yếu tố real trả lời được

Real chỉ cho TỔNG mỗi bin, nên phần của từng người ở bin chung lấy theo tỉ lệ của chính generator
`w[c,b] = contrib[c,b]/Σ`. Từ đó: `zc = z·w` (giá trị real của từng người), `L_c = Σzc/Σcontrib` (**mức**),
`P = zc > 0` (**allele nào đứng**), `s = zc/(L_c·contrib)` (**hình dạng** chiều cao). Mỗi dòng sinh một yếu tố,
hai yếu tố kia lấy đáp án; chép cả ba thì bằng real.

### Cấp 2 — kết quả

Một lỗi phải sửa trước: `w = contrib/Σcontrib` bằng 0 ở bin allele mà generator làm rụng TẤT CẢ người mang, nên
`zc = z·w = 0` và bin mất dù real có đỉnh — tự kiểm thiếu 3.7–5.5 đỉnh. Sửa bằng cách chia đều `z` cho các người
mang ở bin đó, và đọc `L_c` chỉ trên bin cả hai bên đều có đỉnh.

Tự kiểm sau khi sửa khớp về SỐ ĐỈNH (61.8/80.4/91.6/97.5 so với real 61.7/80.4/91.7/97.2) nhưng RFU còn +1…+2 %
vì artefact và nhiễu của generator vẫn cộng lên bin allele đã chép. Nên các dòng được đọc **so với dòng tự kiểm**.

| dòng | gate NOC2/3/4/5 | Δ so tự kiểm | đỉnh allele |
|---|---|---|---|
| tự kiểm (chép cả 3) | +0.134 +0.163 +0.185 +0.125 | — | 61.8 80.4 91.6 97.5 |
| L2-L: sinh **mức** từng người | +0.128 +0.181 +0.215 +0.139 | −0.006 +0.018 +0.030 +0.014 | 61.8 80.3 91.6 97.5 |
| L2-P: sinh **allele nào đứng** | +0.169 +0.180 +0.224 +0.152 | +0.035 +0.017 +0.039 +0.027 | 61.6 82.1 93.0 99.5 |
| **L2-H: sinh HÌNH DẠNG chiều cao** | **+0.338 +0.469 +0.429 +0.187** | **+0.204 +0.306 +0.244 +0.062** | **58.3 77.1 87.3 91.2** |

**Hình dạng chiều cao từng allele trong một người là lỗi**, lớn hơn mức và sự-có-mặt một bậc, và nó giải thích gần
hết độ lệch của L1-A. Dấu hiệu quyết định: L2-H chép đúng real allele nào đứng mà số đỉnh vẫn tụt 3.5–6.3 — hình
dạng của generator đẩy những allele real CÓ xuống dưới ngưỡng phát hiện. Tức phân bố quá tãi, hoặc lệch thấp ở
đầu mờ.

### Cấp 3 — các bước đặt ra hình dạng (việc tiếp theo)

| # | bước | vai trò trong hình dạng |
|---|---|---|
| A6 | `per` = chiều cao quan sát của nguồn / `lam_s` | mang sẵn hình dạng từng allele CỦA NGUỒN |
| A8 | `lam_t ∝ dose · q^(−a·x)` | độ dốc theo kích thước (degradation) trong một người |
| A10 | `HEIGHT_SHARE` | trộn số bản dùng chung theo locus vào `k_h` |
| A11 | `ksum ~ Gamma(k_h/FILT_CV², FILT_CV²)` | **tán xạ chính từng allele** |
| A13 | `sd_al` | jitter từng allele cộng thêm |

### Cấp 3 — tách hình dạng thành từng bước

Giữ nguyên cấu hình L2-H (chép mức + sự-có-mặt + artefact + nhiễu, chỉ sinh hình dạng), tắt lần lượt một bước:

| tắt bước | gate NOC2/3/4/5 | Δ so nền | đỉnh allele |
|---|---|---|---|
| nền L2-H | +0.338 +0.469 +0.429 +0.187 | — | 58.3 77.1 87.3 91.2 |
| A11 tán xạ Gamma (`FILT_CV`) | +0.295 +0.463 +0.438 +0.141 | −0.043 −0.006 +0.009 −0.046 | 58.3 77.4 87.8 91.7 |
| A13 jitter từng allele (`sd_al`) | +0.356 +0.451 +0.435 +0.146 | +0.018 −0.018 +0.006 −0.041 | 58.1 77.4 88.1 91.4 |
| A10 `HEIGHT_SHARE` | +0.350 +0.487 +0.471 +0.163 | +0.012 +0.018 +0.042 −0.024 | 58.4 77.4 87.6 91.2 |
| **A6 `per` phẳng (bỏ mẫu của nguồn)** | +0.267 +0.344 +0.337 +0.150 | **−0.071 −0.125 −0.092 −0.037** | 58.0 77.6 87.9 91.8 |
| **A8 bỏ độ dốc kích thước** | +0.272 +0.433 +0.375 +0.209 | −0.066 −0.036 −0.054 +0.022 | **60.8 79.3 90.3 96.2** |

Hai lỗi khác nhau:

1. **Độ dốc kích thước (A8) gây MẤT ĐỈNH.** Bỏ nó lấy lại gần hết: 91.2 → 96.2 ở NOC5 (real 97.2). Nó đang đẩy
   allele đoạn dài xuống dưới ngưỡng phát hiện. Về cấu trúc thì độ dốc ĐÃ ở dạng phần tăng thêm —
   `h ∝ hs · (eff·t/t_s) · q_tube^(−a_t·x) / q_s^(−a·x)` — nên đây không phải tính hai lần mà là **quá dốc**.
2. **Mẫu từng allele của profile nguồn (A6) gây lệch GATE nhiều nhất** (−0.071/−0.125/−0.092/−0.037): nguồn mang
   theo một mẫu chiều cao riêng của nó mà đích không nên có.

`FILT_CV`, `sd_al`, `HEIGHT_SHARE` đều gần như vô can (|Δ| ≤ 0.046) — nên tán xạ không phải nguyên nhân.

### Cấp 4 — tách lệch chiều cao khỏi rụng, theo kích thước đoạn

Phép đo trước gộp cả đỉnh bị rụng vào trung vị (h = 0 cho log rất âm) nên đọc sai. Tách hai việc: lệch chiều cao
chỉ trên bin CẢ HAI bên đều có đỉnh, và tỉ lệ twin làm rụng allele mà real có.

| | dải bp → | <130 | 130–180 | 180–230 | 230–280 | 280–340 | >340 |
|---|---|---|---|---|---|---|---|
| tự kiểm | lệch log | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| | tỉ lệ rụng | 0.001 | 0.002 | 0.000 | 0.001 | 0.000 | 0.002 |
| nền L2-H, NOC2 | lệch log | −0.108 | −0.060 | +0.032 | +0.146 | +0.184 | +0.166 |
| | tỉ lệ rụng | 0.018 | 0.047 | 0.052 | 0.071 | 0.088 | 0.119 |
| nền L2-H, NOC5 | lệch log | −0.005 | −0.009 | −0.043 | **+0.082** | **+0.112** | +0.089 |
| | tỉ lệ rụng | 0.019 | 0.055 | 0.061 | **0.087** | **0.098** | **0.137** |
| A8 tắt, NOC5 | lệch log | −0.209 | −0.043 | +0.044 | +0.251 | +0.355 | **+0.378** |
| | tỉ lệ rụng | 0.014 | 0.029 | 0.020 | 0.022 | 0.017 | **0.018** |

Hai điều:

1. Twin **cao hơn** real ở đoạn dài (+0.08…+0.18 log) **mà vẫn làm rụng nhiều hơn** ở chính đó (13.7 % so với
   1.9 % ở đoạn ngắn). Trung vị trên real nhưng đuôi dưới dày — **quá tãi**, không phải lệch mức.
2. Bỏ độ dốc kích thước thì **rụng biến mất** (0.137 → 0.018, phẳng theo kích thước) nhưng **lệch chiều cao xấu
   hẳn** (+0.378 ở đoạn dài).

Đây là **một tham số phải làm hai việc**: `lam_t` vừa đặt số bản (quyết định rụng) vừa đặt chiều cao
(`h ∝ per · ksum ≈ per · lam_t`). Real muốn chiều cao giảm MẠNH theo kích thước nhưng sự có mặt giảm NHẸ; một
`a_t` duy nhất không làm được cả hai, nên chỉnh cho chiều cao đúng thì rụng quá nhiều ở đoạn dài và ngược lại.

Nối đúng vào hai luật đã có trong ghi chú dự án (`per-locus-assembly-law`, `generate-then-filter-law`): cả hai đã
khớp SỰ CÓ MẶT trên NOC1 trong ~0.02 ở mọi nơi nhưng CHIỀU CAO thì chưa, và chưa cài. Cấp 4 cho biết vì sao:
chiều cao cần một độ dốc kích thước RIÊNG, dốc hơn độ dốc của sự có mặt.

### Việc tiếp theo

Tách hai độ dốc trong `_filter_contrib`: một số mũ cho số bản (quyết định `k` và do đó sự có mặt) và một số mũ
riêng, dốc hơn, cho RFU trên mỗi bản. Cả hai đo được trên NOC1: đường rụng-theo-kích-thước và đường
chiều-cao-theo-kích-thước là hai phép đo độc lập trên cùng những profile đó.

### Cấp 5 — tách hai độ dốc: `AN_R`, và κ

Cài `AN_R` = phần của độ dốc kích thước tác động lên SỐ BẢN; phần còn lại tác động lên RFU trên mỗi bản, nên
**chiều cao kỳ vọng không đổi** (hai số mũ cộng lại bằng số mũ cũ) còn **sự có mặt** — chỉ phụ thuộc `lam_t` —
thì phẳng hơn. `AN_R = 1` cho lại đúng generator hiện tại. Kiểm tra định danh: `h ∝ hs·(t·eff/t_s)·q_t^(−a_t·x)/q_s^(−a·x)`
giữ nguyên với mọi `AN_R`.

Thêm một nghi phạm mixture-only: `eff = 2^(−κ(k−1))`, κ ~ U(0, 0.26), hạ `lam_t` tới 0.49 lần ở NOC5. Nó **tăng
rụng** nhưng **không hạ chiều cao**, vì bước ngân sách A15 co lại cho khớp tổng — đúng dạng lệch đo được.

| dòng | gate NOC2/3/4/5 | đỉnh allele | rụng NOC5 theo kích thước |
|---|---|---|---|
| tự kiểm | +0.134 +0.163 +0.185 +0.125 | 61.8 80.4 91.6 97.5 | 0.001 → 0.003 |
| nền L2-H | +0.338 +0.469 +0.429 +0.187 | 58.3 77.1 87.3 91.2 | 0.019 → **0.137** |
| κ = 0 | +0.333 +0.437 +0.372 +0.214 | 58.5 77.9 89.0 93.5 | 0.010 → 0.105 |
| AN_R = 0.5 | +0.344 +0.452 +0.436 +0.200 | 60.0 78.7 89.4 94.2 | 0.019 → 0.061 |
| AN_R = 0.25 | +0.331 +0.470 +0.420 +0.241 | 60.3 79.0 89.6 **95.5** | 0.015 → **0.029** |
| AN_R = 0.5 + κ = 0 | +0.339 +0.463 +0.411 +0.257 | 60.1 79.2 90.3 **95.9** | 0.009 → 0.036 |

Phép tách làm đúng việc nó được thiết kế: rụng ở đoạn dài 0.137 → 0.029 và số đỉnh 91.2 → 95.5 (real 97.2), mà
không làm xấu chiều cao. **Nhưng gate gần như không đổi** (+0.338 → +0.331, NOC5 còn xấu hơn). Nên **sự rụng không
phải cái model đọc**; cái nó đọc là **độ nghiêng chiều cao** (twin −0.11 ở đoạn ngắn, +0.18 ở đoạn dài, tức độ dốc
quá phẳng), mà `AN_R` giữ nguyên tổng số mũ nên chỉ giảm nghiêng một phần.

### Cấp 6 — ba giả thuyết về hình dạng, cả ba bị bác

Thêm `A_SCALE` (nhân tổng số mũ kích thước) rồi quét cùng `AN_R`:

| cấu hình | gate NOC2/3/4/5 | đỉnh | nghiêng NOC2 đoạn dài | nghiêng NOC5 đoạn dài |
|---|---|---|---|---|
| nền L2-H | +0.338 +0.469 +0.429 +0.187 | 58.3 77.1 87.3 91.2 | +0.184 | +0.112 |
| a ×1.3, AN_R = 0.5 | +0.375 +0.467 +0.464 +0.244 | 59.2 78.3 88.9 93.4 | +0.099 | +0.004 |
| a ×1.6, AN_R = 0.5 | +0.406 +0.504 +0.477 +0.268 | 58.6 77.8 88.4 92.5 | +0.086 | −0.053 |
| a ×1.6, AN_R = 0.25 | +0.407 +0.502 +0.509 +0.288 | 59.7 78.6 89.4 94.4 | +0.054 | −0.061 |
| a ×2.0, AN_R = 0.25 | +0.445 +0.583 +0.506 +0.278 | 59.2 77.9 88.2 93.2 | +0.043 | −0.138 |

Và tách `per` thành phần hệ thống / riêng của lần chạy, cùng phép "một ống, một mức theo locus":

| cấu hình | gate NOC2/3/4/5 |
|---|---|
| nền L2-H | +0.338 +0.469 +0.429 +0.187 |
| `per` phẳng (đối chứng) | **+0.267 +0.344 +0.337 +0.150** |
| `per` = trung bình các bản lặp (bỏ nhiễu riêng) | +0.330 +0.472 +0.399 +0.185 |
| `per` = hệ thống + rút lại phần riêng | +0.370 +0.474 +0.392 +0.176 |
| một ống, chia chung mức theo locus | +0.335 +0.493 +0.439 +0.141 |

**Ba giả thuyết bị bác bằng đo đạc:**

1. *Độ dốc kích thước gây rụng* — phép tách `AN_R` sửa ĐÚNG sự rụng (0.137 → 0.029 ở đoạn dài, số đỉnh 91.2 → 95.5
   so với real 97.2) nhưng **gate không đổi**. Nên sự rụng không phải cái model đọc.
2. *Độ nghiêng chiều cao là cái model đọc* — làm dốc thêm làm phẳng ĐÚNG độ nghiêng (NOC5 +0.112 → −0.138) nhưng
   **gate xấu đi đơn điệu** (+0.187 → +0.288).
3. *`per` mang nhiễu riêng của lần chạy, hoặc mẫu locus độc lập giữa các người* — trung bình các bản lặp và chia
   chung mức theo locus đều **không giúp**.

Chỉ làm phẳng hẳn `per` là giúp (−0.125), mà nó lại làm chiều cao khớp XẤU HƠN theo từng dải kích thước. Đó là dấu
hiệu chỉ số phản ứng với việc BỚT CẤU TRÚC, không phải với việc giống real hơn — nên đi tiếp theo gate là tự chỉnh
vào chỉ số. Phải chuyển sang tiêu chí từng-con-số trên NOC1 (cân bằng heterozygote, độ tãi trong profile, độ dốc
kích thước).

### Cấp 7 — xét hình dạng bằng tiêu chí từng-con-số trên NOC1

Đo trên 2998 profile NOC1 của train+val, không qua model:

| chỉ số | real | twin | lệch |
|---|---|---|---|
| \|log(h1/h2)\| het, trung vị | 0.2901 | 0.2975 | +2.6 % |
| \|log(h1/h2)\| het, p90 | 0.9823 | 0.9972 | +1.5 % |
| sd(log h) trong profile | 0.5372 | 0.5154 | −4.1 % |
| **độ dốc kích thước, trung vị** | **−0.4912** | **−0.4144** | **−15.6 %** |
| **độ dốc kích thước, sd** | **0.4819** | **0.5483** | **+13.8 %** |

Cân bằng heterozygote và độ tãi trong profile gần như đúng. Hai chỗ sai là **độ dốc kích thước nông hơn 16 %** và
**dao động giữa các profile rộng hơn 14 %** — đo được trên NOC1, không qua model, nên fit được hợp lệ, và nó xác
nhận ĐỘC LẬP phát hiện ở Cấp 4 (twin quá phẳng). Theo `twin-envelope-is-different-extraction` thì tiêu chí
từng-con-số thắng: đây là lỗi đo được, không để lại chỉ vì gate phản ứng ngược chiều.

Quét hệ số trên NOC1:

| cấu hình | độ dốc trung vị | độ dốc sd |
|---|---|---|
| hiện tại | −0.4144 (−15.6 %) | 0.5483 (+13.8 %) |
| a ×1.2 | −0.4686 (−4.6 %) | 0.6287 (+30.5 %) |
| a ×1.4 | −0.5108 (+4.0 %) | 0.7281 (+51.1 %) |
| a ×1.2, AN_R = 0.5 | −0.6034 (+22.8 %) | 0.7575 (+57.2 %) |

Một núm nhân **không thể** sửa cả hai: `a_t = FILT_A[họ]·2^(α + β·log q + s_tube + N(0, sd_d))` là lognormal, nên
nhân vào thì trung vị và độ tãi đi cùng nhau. Và phần tãi thì **đã có sẵn**: profile nguồn là một lần chạy THẬT nên
nó mang một độ dốc degradation thật với đúng độ tãi của real (0.4819); cộng thêm một độ dốc rút ngẫu nhiên lên trên
là tính hai lần — đúng `inherit-then-increment`, và đó là lý do sd đã vượt +13.8 % từ trước.

### Đơn thuốc (hai phần, cả hai đo trên NOC1)

1. **Nâng TRUNG VỊ số mũ** khoảng 1.3× để độ dốc trung vị trúng real −0.4912.
2. **Hạ `sd_d`**, phần rút ngẫu nhiên của `a_t`, vì phần tãi giữa các profile đã nằm trong nguồn. Đích: độ dốc sd
   0.5483 → 0.4819.

Hai mục tiêu độc lập, hai tham số độc lập, cả hai đọc trên NOC1 — không chạm hỗn hợp nào.

### Cấp 8 — phải tách những gì chồng lên bin trước khi đo tỉ lệ

Mọi chỉ số dạng TỈ LỆ chỉ là của allele khi trong bin không có gì khác. Trên NOC1 không có allele của người khác,
nhưng **49.7 % bin allele của một người là đích stutter của một allele khác của chính người đó** (heterozygote hay
nằm cách nhau một hoặc hai repeat), cộng thêm nhiễu nền và pull-up. Đo lại chỉ trên bin sạch: không ở offset
−1/+1/−2/−0.5/−3 của một allele đang đứng khác của cùng người, không có láng giềng khác màu trong 2 bp mang đỉnh
≥ 1000 RFU, và allele đủ cao so với nền (≥ 60 RFU). Với cặp heterozygote thì CẢ HAI bin phải sạch. Cùng luật hai bên.

| chỉ số | mọi bin | **chỉ bin sạch** |
|---|---|---|
| bin dùng/mẫu | 36.0 | 14.5 |
| \|log(h1/h2)\| het, trung vị | +2.6 % | +1.0 % |
| \|log(h1/h2)\| het, p90 | +1.5 % | +5.4 % |
| sd(log h) trong profile | −4.1 % | −2.1 % |
| **độ dốc kích thước, trung vị** | **−15.4 %** | **−2.8 %** |
| **độ dốc kích thước, sd** | +13.1 % | **+18.2 %** |

Khoảng cách 16 % ở trung vị độ dốc **biến mất** (còn 2.8 %): nó là stutter của chính các allele đó và pull-up cộng
lên bin allele, và cộng KHÔNG ĐỀU hai bên vì real nhiều stutter hơn. Real đo trên bin sạch là −0.3557 so với −0.4915
trên mọi bin — nhiễm bẩn làm độ dốc **dốc giả**.

**Nên đơn thuốc "nâng trung vị số mũ 1.3×" ở Cấp 7 là SAI và bị rút.** Cài vào sẽ làm độ dốc quá dốc.

Còn đúng **một** lỗi, và nó RÕ HƠN khi lọc sạch: **độ tãi của độ dốc giữa các profile, +18.2 %** (0.5259 so với
real 0.4448) — đúng câu chuyện `inherit-then-increment`: profile nguồn là một lần chạy thật nên đã mang sẵn một độ
dốc degradation thật với độ tãi của real, rồi generator còn rút thêm một độ dốc ngẫu nhiên lên trên.

### Quét phần rút ngẫu nhiên của `a_t`, trên bin sạch

| `sd_d` × | độ dốc sd | lệch |
|---|---|---|
| 1.0 (hiện tại) | 0.5259 | +18.2 % |
| 0.6 | 0.5019 | +12.8 % |
| 0.3 | 0.4919 | +10.6 % |
| 0.0 (tắt hẳn) | 0.4884 | +9.8 % |

Mọi chỉ số khác giữ nguyên tốt ở mọi giá trị (het trung vị ±2 %, sd(log h) −2 %, độ dốc trung vị −2 %). Nên phần
rút này giải thích khoảng MỘT NỬA phần thừa; 9.8 % còn lại đến từ nguồn khác (ứng viên: mỗi lần chọn một profile
nguồn khác nhau, các số hạng bước template / `AMP_STEP_TILT`).

**Chưa cài, vì NOC1 không tách được cách chia.** Với k = 1 thì `s_tube` (một lần rút cho cả ống) và phần rút riêng
từng người là CÙNG MỘT THỨ, nên NOC1 chỉ ràng buộc TỔNG. Hạ tổng là hợp lệ; dồn phần hạ vào đâu thì không đo được:

- dồn vào **phần từng người** → mọi người trong một ống chung một độ dốc, mà `cross-locus-ranking-lever` nói
  per-contributor β_c là tín hiệu thật và hữu ích
- dồn vào **`s_tube`** → giữ khác biệt giữa người nhưng mất khác biệt giữa các ống

Theo "thà không có luật còn hơn sai luật": tìm nốt nguồn 9.8 % còn lại trước, vì nếu đó là việc chọn profile nguồn
thì `sd_d` có thể không cần hạ chút nào.

### Trạng thái các tham số chẩn đoán

`AN_R`, `A_SCALE`, `sd_d` scale, `per_sys`, `per_tube` hiện CHỈ có trong bản chẩn đoán `scratchpad/mi_stage.py`,
**chưa cài** vào `make_insilico.py`. Trong số đó `AN_R` là cái duy nhất đã chứng minh là cải thiện thật theo tiêu chí
từng-con-số (rụng ở đoạn dài 13.7 % → 2.9 %, số đỉnh allele 91.2 → 95.5 so với real 97.2) dù không đổi gate.

### Cấp 9 — sd còn bị chồng lấn hai thứ nữa

**Chồng lấn 1: sai số của chính phép ước lượng.** Mỗi độ dốc là một hồi quy trên ~14 bin sạch nên nó có sai số
riêng; sd quan sát = √(sd thật² + se²). Và se là phần LỚN NHẤT:

| | real | twin | lệch |
|---|---|---|---|
| sd quan sát (đã trừ treatment × template) | 0.2626 | 0.3772 | +43.6 % |
| se trung bình của phép ước lượng | 0.2164 | 0.2254 | |
| **sd THẬT** | **0.1488** | **0.3024** | **+103.3 %** |

Độ tãi độ dốc của twin **gấp đôi** real, không phải hơn 18 % như sd thô cho thấy. Nhiễu ước lượng che gần hết.

**Chồng lấn 2: phân tầng mà tên file cho.** 99.8 % ô (donor, treatment, template, dilution) là MỘT sản phẩm PCR và
các file trong ô là các lần INJECT của ống đó. Nên NOC1 tách được đúng ba tầng — điều t đã sai khi nói không tách được:

| thành phần | real | twin |
|---|---|---|
| **injection (trong cùng một ống)** | **0.0000** | **0.1537** |
| ống (giữa các ô cùng người) | 0.2252 | 0.2830 |
| người (giữa các người) | 0.1028 | 0.1678 |

Real có độ dốc **giống nhau tuyệt đối** giữa các lần inject cùng một ống — đúng như convention nói. Twin bịa ra
0.154 vì bộ thử rút nguồn và rút `a_t` MỚI cho từng file. Đó là lỗi của BỘ THỬ, không phải của generator: trong dữ
liệu huấn luyện mỗi mẫu là một ống riêng nên không có bản lặp injection nào.

(Lưu ý: chỉ tổng và thành phần injection được trừ se; hai thành phần ống/người chưa trừ nên giá trị tuyệt đối của
chúng còn bị phồng.)

### Rút mỗi ô, và `sd_d` bị bác

Rút một lần cho mỗi ô (thay vì mỗi file) đưa thành phần injection của twin về **đúng 0 như real**, nhưng **tổng
không đổi** — phương sai chỉ dời sang tầng ống:

| thành phần | real | twin (rút mỗi file) | twin (rút mỗi ô) |
|---|---|---|---|
| **sd THẬT** | **0.1488** | 0.2958 (+98.8 %) | **0.2976 (+100.0 %)** |
| injection | 0.0000 | 0.1396 | **0.0000** |
| ống | 0.2252 | 0.2802 | 0.3250 |
| người | 0.1028 | 0.1637 | 0.1674 |

Quét `sd_d` bằng thước đã hiệu chỉnh (trừ se, rút mỗi ô):

| `sd_d` × | sd THẬT của twin | lệch |
|---|---|---|
| 1.0 | 0.2982 | +100.5 % |
| 0.5 | 0.3052 | +105.2 % |
| 0.0 | 0.2933 | +97.2 % |

Ba giá trị không khác nhau: **`sd_d` KHÔNG phải nguyên nhân**, và đơn thuốc "hạ `sd_d`" bị rút. Lần quét trước thấy
nó giải thích một nửa chỉ vì thước khi đó còn chứa cả hai chồng lấn.

**Chốt:** độ tãi độ dốc kích thước của twin **đúng gấp đôi real** (0.2982 so với 0.1488, phương sai gấp 4), sau khi
đã trừ sai số ước lượng và đã rút một lần cho mỗi ống. Đây là lỗi duy nhất còn sống sót của tầng allele theo tiêu
chí từng-con-số.

### Ứng viên tiếp theo cho phần gấp đôi đó

| ứng viên | lý do |
|---|---|
| `AMP_STEP_TILT` | chú thích của chính nó: "độ dốc kích thước phẳng dần theo template" — một phép đổi độ dốc có rút riêng |
| `_INJ_STEP` | hệ số bước injection, rút riêng mỗi lần |
| chọn profile nguồn | mỗi twin lấy một lần chạy NOC1 khác nhau, nên đã mang sẵn một độ tãi bằng của real; cộng thêm bất kỳ độ dốc mô hình nào lên trên là làm phương sai gấp lên |

Ứng viên thứ ba đủ để giải thích một phần √2 nhưng không đủ cho 2×, nên phải đo cả ba.

### Cấp 10 — hỏi pipeline thay vì đoán ứng viên

Trong `_filter_contrib` chiều cao là một TÍCH, nên `log h` là một TỔNG và độ dốc theo kích thước là tổng độ dốc của
từng thừa số:

```
log h = log hs − log lam_s − log _es + log _et + log ksum + log f + log _jit + log _alj
```

Nên `Var(độ dốc)` phân rã được theo đúng từng bước. Trước hết, chỉ bằng cách đọc code: `AMP_STEP_TILT` và `_lvl_var`
**chỉ dùng ở đường cũ** (dòng 1817), không dùng trong `_filter_contrib` — loại một ứng viên mà không cần đo.

Ghi từng thừa số trong MỘT lượt chạy, độ dốc fit trên chính các bin sạch (2485 profile NOC1):

| thành phần | độ dốc TB | sd | phần phương sai của TỔNG |
|---|---|---|---|
| `hs` (profile nguồn thật) | −0.0491 | 0.1633 | +6.5 % |
| `−lam_s` | +0.0523 | 0.1416 | −2.1 % |
| `−_es`, `+_et` | 0 | 0 | 0 |
| **`ksum`** | **−0.4945** | **0.5374** | **+95.6 %** |
| `alj` | 0.0008 | 0.0166 | −0.0 % |
| TỔNG | −0.4905 | 0.5445 | 100 % |

Kiểm định danh khớp tuyệt đối: sd(TỔNG) = sd(tổng các phần) = 0.5445.

**Hai điều:**

1. **`ksum` mang 95.6 % phương sai** — toàn bộ độ dốc (−0.4945 trên tổng −0.4905) và độ tãi của nó đến từ mô hình
   số bản `lam_t`, không từ dữ liệu.
2. **Thông tin thật của nguồn bị triệt tiêu.** `hs` (−0.049) và `−lam_s` (+0.052) gần như bù trừ hết nhau, r = −0.609.
   Generator **chia bỏ** độ dốc kích thước thật của profile nguồn rồi **áp lại một cái mô hình** — nên cả trung vị
   lẫn độ tãi đều là của mô hình. Đây là lý do gốc mà `sd_d` vô can: `sd_d` chỉ là một mảnh nhỏ của `a_t`, còn toàn
   bộ độ dốc thì do `lam_t` dựng lên từ đầu.

### Luật hay nhiễu: `lam_t` không có nhiễu nên tách được

| | độ dốc TB | sd |
|---|---|---|
| **`lam_t` (LUẬT, không nhiễu)** | **−0.6627** | **0.6734** |
| nhiễu (`ksum − lam_t`) | +0.1682 | 0.3452 |
| tổng cuối | −0.4905 | 0.5445 |
| **real (đã trừ se)** | −0.3557 | **0.1488** |

**Bước sai là `lam_t`**, và sai cả hai chiều: độ dốc trung bình −0.663 so với real −0.356 (dốc gần gấp đôi) và độ tãi
0.6734 so với 0.1488 (**gấp 4.5 lần**).

**Và lại là hai lỗi triệt tiêu nhau** — cùng mẫu với cặp n−1 / nhiễu nền ở Phần 5. Nhiễu Poisson + Gamma lệch hệ
thống +0.168 (Jensen: log của số đếm nhỏ bị kéo xuống, mạnh hơn ở đoạn dài nơi `lam_t` nhỏ, nên nó LÀM PHẲNG độ
dốc), kéo −0.663 về −0.49, và trên bin sạch ra −0.346 so với real −0.356. **Trung vị đúng nhờ hai cái sai bù nhau**;
độ tãi không bù hết nên lộ ra gấp đôi.

Đó cũng là lý do `sd_d` vô can: nó chỉ là một mảnh của `a_t`, còn cả độ dốc do `lam_t` dựng từ đầu. Nguồn của độ tãi:
`log lam_t` có độ dốc `−a_t·ln(q_tube)` trong khi `a_t = FILT_A·2^(al_s + be_s·log q_tube + s_tube + N(0,sd_d))` —
một TÍCH của hai số hạng cùng phụ thuộc q, nên độ tãi của q bị khuếch đại.

### Cảnh báo cho bước sửa

Không được sửa một mình. Hạ độ tãi của `lam_t` mà giữ nguyên trung bình sẽ **phá trung vị**, vì trung vị hiện đúng
nhờ nhiễu Poisson + Gamma bù +0.168. Phải chỉnh cùng lúc: độ dốc của luật về quanh real −0.356 **cộng** phần bù của
nhiễu, và độ tãi về 0.1488. Cả hai đo được trên NOC1 bằng đúng thước này (bin sạch, trừ se, rút một lần mỗi ô).

### Cấp 11 — đo hai mục tiêu cùng lúc

Thêm hai núm tách biệt cho luật `lam_t`, vì độ dốc của nó là `−a_t·ln(q_tube)`:
`a_t = FILT_A · A_SCALE · 2^(SHRINK·(al_s + be_s·log q + s_tube + N(0,sd_d)))`. `SHRINK` = 0 bỏ hết phần lệch log
của `a_t`; `A_SCALE` đặt mức.

| cấu hình | luật TB | luật sd | nhiễu | tổng TB | tổng sd THẬT |
|---|---|---|---|---|---|
| **MỤC TIÊU = REAL** | — | — | — | **−0.3557** | **0.1488** |
| hiện tại | −0.6149 | 0.3981 | +0.1458 | −0.4630 | 0.2871 |
| shrink 0.5 | −0.6094 | 0.3926 | +0.1560 | −0.4492 | 0.2749 |
| shrink 0.0 | −0.6055 | **0.4471** | +0.1662 | −0.4372 | 0.3000 |
| shrink 0.0, a ×0.8 | −0.5047 | 0.3766 | +0.1275 | −0.3780 | 0.2501 |
| shrink 0.0, a ×0.6 | −0.3913 | 0.2989 | +0.0906 | −0.3092 | **0.1929** |
| shrink 0.5, a ×0.8 | −0.4900 | 0.3195 | +0.1143 | **−0.3765** | 0.2159 |

1. **`SHRINK` không giảm độ tãi của luật** (0.3981 → 0.3926 → **0.4471**, còn tăng). Nên toàn bộ phần lệch log của
   `a_t` — gồm `sd_d`, `s_tube` và hệ số họ treatment — **không phải** nguồn của độ tãi. Bác thêm một ứng viên.
2. **`A_SCALE` kéo trung bình và độ tãi đi cùng nhau, gần tỉ lệ thuận.** Ở chỗ trung bình khớp real (a ×0.8:
   −0.3765 so với −0.3557) độ tãi vẫn **+45 %**; ở chỗ độ tãi gần nhất (a ×0.6: 0.1929) trung bình lại nông quá.

Hai núm không tách được vì cùng nằm trong một tích: độ tãi của `−a_t·ln(q_tube)` chủ yếu do **`ln(q_tube)` khác nhau
giữa các ống**, mà q là do spec và khớp real. Nên phát biểu đúng là: **độ dốc thật của real không phụ thuộc q mạnh
như luật nói.**

### Cấp 12 — hồi quy độ dốc theo log q: luật thiếu một số hạng

Hồi quy độ dốc kích thước của từng profile theo `log q_tube`, cùng profile cùng bin cùng phép ước lượng hai bên:

| | hệ số theo log q | dư sd |
|---|---|---|
| **REAL** | **−0.1592** | 0.4130 |
| **LUẬT (log lam_t)** | **−0.4650** | 0.4828 |
| tổng của twin (nhiễu pha loãng) | −0.2980 | 0.4488 |
| luật sau a ×0.8 | −0.4261 | 0.3407 |

**Luật cho độ dốc phụ thuộc q mạnh gấp 2.9 lần real.** Và đó là lỗi CẤU TRÚC, không chỉnh được bằng hệ số: để hệ số
về −0.159 cần a ×0.34, nhưng khi đó trung bình độ dốc chỉ còn −0.22 so với real −0.356.

**Luật thiếu một số hạng không phụ thuộc q.** `lam_t ∝ q_tube^(−a_t·x)` cho độ dốc `−a_t·ln(q_tube)`, nên ở ống không
hư hại (q = 1) nó cho độ dốc **bằng 0**, còn real vẫn dốc đáng kể. Về sinh học thì hiển nhiên: **PCR khuếch đại
amplicon dài kém hơn ngay cả với DNA nguyên vẹn**, độc lập với degradation. Thiếu số hạng đó nên luật phải gánh cả
phần nền bằng kênh q — kênh q quá mạnh, và vì q khác nhau giữa các ống nên độ tãi phồng lên gấp đôi. Trung bình vẫn
đúng chỉ vì phần quá dốc đó lại được nhiễu Poisson làm phẳng trở lại.

### Đơn thuốc, hai số đo được trên NOC1

Viết độ dốc kích thước của real thành `A + B·log q` với **B = −0.159** đo trực tiếp, và **A** giải từ trung bình
độ dốc −0.3557 cùng trung bình `log q` của tập. Trong generator:

1. thêm một suy giảm theo kích thước **không phụ thuộc q** (hiệu suất PCR theo chiều dài amplicon), đặt mức A
2. hạ số mũ theo q để hệ số về **−0.159** thay vì −0.465

Cả hai đọc bằng đúng một phép hồi quy trên NOC1, và đặt vào **hiệu suất RFU trên mỗi bản** (không phải vào số bản)
thì sự có mặt không bị chạm — tức dùng đúng bậc tự do `AN_R` đã dựng ở Cấp 5.

### Cấp 13 — thử cài, KHÔNG đạt, đã hoàn nguyên

Cài `FILT_QSLOPE` / `FILT_QBASE` / `FILT_QJENSEN`: triệt tiêu độ dốc của `per` và của luật `lam_t` trong CHIỀU CAO
(để nguyên số bản, nên sự có mặt không bị chạm) rồi áp `A + B·log q_tube`. Năm lượt, mỗi lượt lộ một khớp nối:

| lượt | sai ở đâu |
|---|---|
| 1 | dùng phần tăng thêm `ln q_đích − ln q_nguồn`, nhưng q của nguồn không nhỏ như giả định → tổng −0.063 |
| 2 | đo độ dốc nguồn từ `hs` thay vì giả định, vẫn sót số hạng `+a_nguồn·ln q_nguồn` của `per` → −0.106 |
| 3 | cộng lại số hạng đó → −0.132, vẫn xa |
| 4 | `per` dùng **trung vị** ở bin nguồn không có đỉnh (phần lớn là đoạn dài) nên `per` phẳng hơn `hs`; đo trực tiếp độ dốc của `per` → −0.132 |
| 5 | phát hiện t đã **đoán** `mean(log q) = 2.0` để suy ra `A`; đo thật: `A = −0.3259`, `B = −0.1520`, `mean(log q) = 0.6778` |

Sau lượt 5:

| | tổng TB | tổng sd THẬT | hệ số q |
|---|---|---|---|
| **real** | −0.3557 | **0.1488** | **−0.1585** |
| trước khi cài | −0.4630 | 0.2871 (+93 %) | −0.2980 |
| sau khi cài | **−0.3734** ✓ | **0.0688 (−54 %)** | **−0.0256** |

Trung bình đạt, nhưng độ tãi và hệ số q **hỏng theo chiều ngược lại**. Lý do là **một lỗi chân đế nữa, ở phép ước
lượng**: A và B được đo trên ĐẦU RA (độ dốc của các đỉnh đã vượt ngưỡng) rồi đem áp làm luật ĐẦU VÀO. Đầu ra bị
kiểm duyệt — luật càng dốc thì đoạn dài càng rụng nên độ dốc đo được phẳng lại — nên áp một luật đã-bị-kiểm-duyệt
rồi kiểm duyệt lần nữa cho ra quá phẳng. Thêm vào đó độ lệch Jensen của Poisson **tương quan với q**, nên nó triệt
tiêu một phần chính số hạng `B·log q` vừa áp (hệ số đo được 0.026 trên 0.152 áp vào, tức chỉ 17 % truyền qua).

**Đã hoàn nguyên `make_insilico.py` về `gen_bak32.py`** (trạng thái sau các bản sửa nhiễu nền, trước bản cài này).
Giữ lại toàn bộ số đo.

### Cách cài đúng, cho lượt sau

Không fit trên đầu ra rồi áp vào đầu vào. Phải giải **ngược qua phép kiểm duyệt**: chọn `(A′, B′)` của luật sao cho
ĐẦU RA của twin khớp đầu ra của real. Đo hệ số truyền bằng hai điểm (đã có: nền áp −0.047 → ra −0.132; áp −0.3259 →
ra −0.3734, tức hệ số truyền ≈ 0.87 cho phần nền, nhưng chỉ ≈ 0.17 cho kênh q) rồi giải; hoặc đơn giản hơn, quét
`(A′, B′)` trên NOC1 và chọn điểm mà **cả ba** số khớp: trung bình, độ tãi thật, và hệ số theo log q. Hệ số truyền
của kênh q thấp bất thường (0.17) nên phải kiểm nó trước khi tin bất kỳ nghiệm nào.

### Cấp 14 — cơ chế lấy mẫu, và số hạng thiếu thực sự đáng bao nhiêu

**Cơ chế lấy mẫu** (`_filter_source`): `pool_lvl` có 45 người, mỗi người 6–7 mức template; hàm lấy **mức cao nhất**
có một lần chạy ở injection ≥ đích, ưu tiên đúng injection, rồi trả về ngay. Đo trên 1500 lượt:

| | |
|---|---|
| q của **nguồn** | trung vị **1.000**, p90 1.100 → nguồn gần như KHÔNG hư hại |
| q của **đích** | trung vị **1.500**, p90 **9.730** |
| tương quan log q nguồn vs đích | **r = +0.053** → **không ghép theo q** |
| template nguồn / đích | trung vị **4.0×**, p90 **32×** |
| bin nguồn không có đỉnh (phải thay trung vị) | **0.1 %** |

Nên generator ngoại suy rất xa: lấy một lần chạy sáng, nguyên vẹn rồi chế tạo ra một mẫu mờ, hư hại. **Toàn bộ
degradation là mô hình, không kế thừa gì** — đó là lý do mọi sai của luật độ dốc hiện ra thành toàn bộ độ dốc, và
`FILT_A["a"] = 2.0` phải gánh hết.

**Số hạng thiếu**: đẳng thức đúng nhưng nhỏ.

| | |
|---|---|
| độ dốc log `hs` | −0.0771 |
| độ dốc log `per` | −0.0371 |
| hiệu | +0.0400 |
| `a·log(q_s)` dự báo | +0.0364 ✓ (dư +0.0036) |

Chia cho `lam_s` **cộng** `+a·log(q_s)` độ dốc, tức gỡ suy giảm mô hình của nguồn — nhưng chỉ đáng **+0.036** vì
q nguồn ≈ 1. Và phép thay trung vị chỉ ở **0.1 %** bin, nên **lý do nêu ở lượt 4 là SAI**, không đáng kể. Lượt 2, 3,
4 đều đuổi theo những số hạng ≤ 0.04 trong khi lỗi thật là hằng số bị ĐOÁN (−0.047 thay vì −0.3259, lệch 0.28).

**Và điều quan trọng hơn cả:** nguồn có q ≈ 1 nhưng độ dốc chỉ **−0.077**, còn phép hồi quy nói ở q = 1 độ dốc phải
là **−0.326**. Hai số mâu thuẫn ⇒ phép hồi quy `độ dốc = A + B·log q` **bị nhiễu bởi template** (ng 0.0078 → −0.131,
ng 0.25 → −0.467). Nên `A` KHÔNG phải "độ dốc của ống nguyên vẹn", và đó là lý do gốc bản cài hỏng: fit một đường hai
tham số lên một quan hệ bị nhiễu rồi đem áp làm luật sinh. Lượt sau phải fit **template và q cùng lúc** (và họ
treatment), tức đúng cái mà `FILT_S`/`FILT_A` vốn định làm.

### Cấp 15 — độ dốc của real: phân tầng đúng thì hết mâu thuẫn

Năm con số mâu thuẫn ở Cấp 14 đều đến từ nhiễu. Đo lại có giữ biến gây nhiễu cố định:

**`pool_lvl` KHÔNG bị làm phẳng** — nó là `Xu = Xflat_train[ss]` với `clean = (lab_ == "a")`, tức real gốc, untreated,
train. Ghép đúng từng mức template thì hai tập khớp:

| ng | real (Xflat) | pool_lvl |
|---|---|---|
| 0.0078 | −0.053 | −0.044 |
| 0.0625 | −0.063 | −0.047 |
| 0.2500 | −0.093 | −0.096 |
| 0.5000 | −0.027 | −0.023 |

Kết luận "pool phẳng hơn 2–3 lần" là **lỗi ghép của tôi** (phân vị 80 % template trên hai phân bố khác nhau, và
`Xflat` còn chứa mẫu treated). **Rút.**

**Real untreated: độ dốc ≈ −0.063, và KHÔNG phụ thuộc template** (−0.019 … −0.176 qua bảy mức, không xu thế). Trong
untreated, q không biến thiên (= 1.000).

**Toàn bộ độ dốc đến từ TREATMENT, không phải q:**

| họ | độ dốc | q trung vị |
|---|---|---|
| a (untreated) | **−0.063** | 1.000 |
| b | −0.197 | 1.000 |
| c | −0.400 | 1.100 |
| d | −0.585 | 1.600 |
| e | −0.680 | 2.500 |
| I15 / I22 (humic) | −0.018 / −0.019 | 1.0 / 1.4 |
| −30 / −45 (DNase) | −0.548 / −0.494 | 1.7 / 2.7 |
| U60 / U105 (UV) | −0.512 / −0.432 | 13.3 / 20.3 |
| **S30 (sonication)** | **−1.587** | 4.9 |

Humic có q tới 1.4 mà độ dốc gần 0, còn sonication S30 có q 4.9 với độ dốc −1.59: **q không điều khiển độ dốc**, nó
chỉ là biến đại diện cho treatment. Nên "độ dốc của real" không phải một số — nó là −0.06 cho ống nguyên vẹn tới
−1.59 cho ống bị phá nặng.

**Rút `A = −0.3259`.** Con số đó là trung bình trộn treatment, không phải "độ dốc của ống nguyên vẹn" — giá trị đúng
là **−0.063**. Áp nó như luật nền là lý do bản cài vượt quá. Và `−0.077` của profile nguồn thì **đúng** và khớp
untreated, không hề mâu thuẫn.

### Hệ quả cho lượt sau

Luật độ dốc phải theo **họ treatment và liều**, không theo q — đúng cái `FILT_S` / `FILT_A` vốn được khoá theo họ
treatment. Cần đo thêm: độ tãi của độ dốc **trong từng họ** (real), vì chính nó mới là mục tiêu, thay cho con số
0.1488 trộn lẫn đang dùng.

## Phần 8 — danh sách việc, cập nhật sau mỗi lần sửa

Gate là điểm cuối và nó **trộn được nhiều lỗi ngược dấu để bù nhau** (đã gặp ba lần trong phiên: n−1 thiếu bù nhiễu
thừa; luật độ dốc quá dốc bù nhiễu Poisson làm phẳng; và lần này). Nên danh sách việc chỉ dùng bảng từng-con-số.

### Mục 3 (`AN_R`) — BỊ BÁC, không cài

`AN_R` tách số mũ kích thước thành phần-số-bản và phần-RFU-mỗi-bản, nên nó điều khiển đúng **đường rụng theo kích
thước**. Fit nó trên NOC1 — tập được phép — bằng tỉ lệ allele của panel thực sự đứng, theo dải kích thước:

**template ≤ 0.03 ng**

| | <130 | 130–180 | 180–230 | 230–280 | 280–340 | >340 | \|lệch\| TB |
|---|---|---|---|---|---|---|---|
| REAL | 0.7716 | 0.7007 | 0.6286 | 0.5522 | 0.4951 | 0.4588 | |
| **AN_R = 1.00 (hiện tại)** | 0.8021 | 0.7142 | 0.6436 | 0.5674 | 0.4892 | 0.4533 | **0.0143** |
| AN_R = 0.50 | 0.8186 | 0.7755 | 0.7319 | 0.6845 | 0.6328 | 0.5832 | 0.1032 |
| AN_R = 0.25 | 0.8372 | 0.8151 | 0.7765 | 0.7477 | 0.7186 | 0.6508 | 0.1565 |
| AN_R = 0.00 | 0.8425 | 0.8253 | 0.8231 | 0.8041 | 0.7756 | 0.6809 | 0.1907 |

template > 0.03 ng: AN_R = 1.00 cho |lệch| **0.0099**; 0.5 / 0.25 / 0.0 cho 0.0435 / 0.0524 / 0.0561.

**Luật sự-có-mặt theo kích thước của generator đã đúng gần như tuyệt đối**, và hạ `AN_R` phá nó. `AN_R` chỉ tỏ ra tốt
khi đo bằng thước phía hỗn hợp ở Cấp 4–5 — nhưng thước đó là bộ thử L2-H, nơi sự-có-mặt của real **bị chép vào (ép
đứng)** rồi chiều cao do generator sinh, nên allele bị đẩy xuống dưới ngưỡng thì rụng. Đó là hiệu ứng **CHIỀU CAO**,
bị gán nhãn sai thành "rụng do luật sự-có-mặt". Mục 3 **xoá khỏi danh sách**.

### Mục 1 (nhiễu nền quá cao) — ĐẠT, và nó là hệ quả của bản sửa trước

Đo lại trên NOC1 thì nhiễu **không "chạy cao"**: phân bố theo dải gần khớp, dải cao chỉ lệch +0.01…+0.02 đỉnh, và
tắt `NOISE_GAIN` (+4.5 %) hay tắt tỉa đám đông (+6.8 %) đều không giúp. `NOISE_GAIN` cũng không phải ngoại suy: dải
dư lượng của NOC1 (−1.30…1.12) và của hỗn hợp (−1.42…0.94) **trùng nhau**, và chiều cao trung vị của nhiễu real gần
như **phẳng** theo dư lượng (8–9 RFU suốt dải; phản ứng thật NOC1 +0.056, hỗn hợp +0.009, còn `NOISE_GAIN` giả định
+0.070).

Chỗ thật nằm ở **ĐUÔI**, và chỉ ở hỗn hợp. Đếm đỉnh ở bin trống theo dải (allele + artefact chép từ real):

| dải | 25–40 | 40–70 | 70–150 | >150 |
|---|---|---|---|---|
| REAL NOC3 | 0.06 | 0.02 | 0.01 | 0.00 |
| twin NOC3 (trước) | 0.13 | 0.09 | 0.05 | 0.01 |
| REAL NOC4 | 0.09 | 0.02 | 0.00 | 0.00 |
| twin NOC4 (trước) | 0.19 | 0.14 | 0.04 | 0.01 |

Drop-in không phải nguyên nhân (tắt nó NOC3 vẫn +22.2 %).

**Nguyên nhân là hệ quả của bản sửa `_CR_MAX` trước đó.** Dải cao (h ≥ 25) của bảng đám đông chạy 0.155 → **4.077**
qua các cột; tôi đã xác định hàng đó là **độ sáng, không phải đám đông** (pull-up trên NOC1 sáng) nhưng chỉ dùng nhận
xét đó để chặn `_CR_MAX`, **vẫn để bảng tra theo cột**. Hỗn hợp có 100+ đỉnh cao nên đọc cột cuối = 4.077, rồi
`_keep = min(4.077/1.279, 1) = 1` → đỉnh nhiễu cao **không bị tỉa gì cả**; hỗn hợp thừa hưởng pull-up của NOC1 sáng.

Sửa: các dải cao (`CR_TALL_FROM = 3`, tức h ≥ 16) **giữ ở giá trị cột thưa**, không tra theo cột. Kết quả:

| | NOC1 | NOC2 | NOC3 | NOC4 | NOC5 |
|---|---|---|---|---|---|
| nhiễu, số đỉnh | +0.77 → **+0.52** | +0.19 → **−0.22** | +0.49 → **+0.42** | +0.79 → **+0.29** | +1.31 → **+0.28** |
| nhiễu, RFU | +4.9 % → **+2.3 %** | +11.5 % → **+3.4 %** | +20 % → **+13.3 %** | +16 % → **+1.4 %** | +1.3 % → **−7.0 %** |
| tổng đỉnh | +0.60 → **+0.26** | −0.90 → −0.78 | +2.19 → **+2.06** | +2.77 → **+2.27** | +4.18 → **+2.93** |

Đỉnh nhiễu 70–150 RFU ở bin trống: twin 0.04–0.07 → **0.01–0.02** (real 0.00–0.01).

**Và nó làm lộ một cặp triệt tiêu nữa**: n−2 RFU −14.0 % → −16.6 %, n−3 −5.1 % → −9.7 %, vì nhiễu quá cao trước đây
rơi cả vào bin ở vị trí n−2/n−3 và bù cho phần thiếu thật của hai lớp đó.

### Mục 2 — KHÔNG phải lỗi tỉ lệ, và không phải năm lỗi mà là MỘT

Số đỉnh của n−2 / n−0.5 / n−3 gần đúng (−0.15 / −0.06 / −0.05) mà RFU thiếu (−16.6 / −23.1 / −9.7 %) nên nghi là tỉ
lệ chiều cao. Đo tỉ lệ trên **vị trí sạch một-nguồn** (đích của đúng MỘT allele đang đứng, không là đích của offset
nào khác, không là bin allele, không cạnh pull-up), thang UNITS của bộ phân loại:

| offset | tỉ lệ real | tỉ lệ twin | lệch | cha real | cha twin |
|---|---|---|---|---|---|
| n−1 | 0.0689 | 0.0725 | +5.3 % | 684 | **606** |
| n+1 | 0.0219 | 0.0205 | −6.6 % | 863 | **728** |
| n−2 | 0.0141 | 0.0151 | **+7.2 %** | 744 | **555** |
| n−0.5 | 0.0493 | 0.0827 | **+67.7 %** | 156 | **104** |
| n−3 | 0.0309 | 0.0402 | **+30.2 %** | 272 | **219** |

Tỉ lệ của twin **cao hơn**, không thấp — và ghép theo dải chiều cao cha thì tỉ lệ khớp tốt ở mọi dải. Cái sai là
**phân bố cha**: tích ra n−2 real 0.0141 × 744 = 10.5 RFU so với twin 0.0151 × 555 = 8.4 (−20 %), khớp phần thiếu.

Xác suất artefact HIỆN theo chiều cao cha (real / twin):

| offset | 0–100 | 100–250 | 250–500 | 500–1k | 1k–2k | >2k |
|---|---|---|---|---|---|---|
| n−1 | 0.143/0.144 | 0.334/0.362 | 0.582/0.585 | 0.798/**0.720** | 0.876/**0.778** | 0.881/**0.788** |
| n+1 | 0.086/0.090 | 0.092/0.096 | 0.109/0.118 | 0.137/0.168 | 0.215/0.240 | 0.439/**0.399** |
| n−2 | 0.088/0.095 | 0.087/0.087 | 0.097/0.088 | 0.116/0.109 | 0.160/0.175 | 0.397/**0.321** |
| n−0.5 | 0.055/0.052 | 0.041/0.036 | 0.033/0.027 | 0.033/**0.022** | 0.030/**0.016** | 0.041/**0.019** |
| n−3 | 0.090/0.095 | 0.084/0.092 | 0.078/0.084 | 0.078/0.086 | 0.093/0.081 | 0.146/**0.101** |

**Ở cha sáng, twin thiếu artefact nhìn thấy được ở MỌI offset** — một cơ chế chung, không phải năm luật riêng. Đây
cũng chính là phần dư của bản sửa n−1 ở Cấp 3 (còn −0.38 đỉnh), giờ thấy là chung cho cả năm lớp.

**Mâu thuẫn số học chỉ đúng nguyên nhân:** real cho xác suất hiện ở cha > 2k là **0.40 / 0.44 / 0.15** cho
n−2 / n+1 / n−3, trong khi `rate` của `art_table` chỉ là **0.088 / 0.077 / 0.051**. Không thể hiện nhiều hơn phát.
Nên **`rate` đang là tỉ lệ NHÌN THẤY trộn lẫn, không phải tỉ lệ PHÁT**: với offset có tỉ lệ chiều cao nhỏ, phần lớn
lần phát nằm dưới ngưỡng ở cha mờ nên phép fit gộp cho ra con số thấp giả. Riêng n−1 có tỉ lệ lớn nên hai cái gần
trùng — đó là lý do chỉ n−1 có rate đúng.

**Đơn thuốc:** dẫn lại `rate` từng offset như **tỉ lệ PHÁT**, đo trên **cha sáng** nơi khả năng nhìn thấy ≈ 1, thay
cho phép gộp mọi cha. Một thay đổi, phủ cả năm lớp, fit được trên NOC1.

### Mục 2′ — mở pipeline của chính chỉ số đó, rồi ghi sổ từng bước

Giả thuyết "`rate` là tỉ lệ nhìn thấy trộn lẫn" **SAI**, và sai vì lấy sai số: nó dựa trên `rate` **gộp** của
`art_table` (n−2 0.088, n+1 0.077) trong khi code dùng `art_table_loc`, **theo locus**, cao hơn nhiều. Thay vì đổi
sang một giả thuyết khác, mở pipeline của đúng chỉ số "cha sáng có artefact nhìn thấy được không" — S1…S12 ở Phần 7 —
rồi ghi sổ từng bước trong MỘT lượt chạy (real chỉ cho kết quả cuối nên không chép đáp án từng bước được):

| offset | n cha > 2k | rate dùng | phát | **sống / đã phát** | twin hiện | REAL hiện |
|---|---|---|---|---|---|---|
| n−1 | 4896 | 0.8027 | 0.8049 | **0.9604** | 0.7919 | 0.8807 |
| n+1 | 5163 | 0.4754 | 0.4796 | **0.7116** | 0.4007 | 0.4432 |
| n−2 | 5101 | 0.5430 | 0.5417 | **0.4879** | 0.3221 | 0.3972 |
| n−3 | 5508 | 0.0473 | 0.0450 | 0.4032 | 0.0986 | 0.1462 |

**Bước sai là S10 (survival)**: n−1 sống 0.960 nhưng n−2 chỉ 0.488 và n+1 0.712, trong khi ở cha sáng thì đáng lẽ gần
1. Cơ chế: `ART_SURV_P` tra theo chiều cao THỰC HIỆN của artefact, và offset yếu có tỉ lệ nhỏ nên ở cha 2000 RFU
chúng chỉ ~30 RFU, rơi vào vùng survival 0.7–0.87 rồi còn nhân `_CARR_MULT`. Và `calibrate.py` nói rõ đường đó fit
**chỉ bằng n−1** ("n-1 ALONE. Pooling the four product offsets was tried and biases the curve down") — tức một đường
của n−1 đang được áp cho mọi offset.

**Một lỗi đo phải sửa trước khi đọc dòng n−3:** điều kiện "sạch một-nguồn" chỉ loại 9 offset theo thang bộ phân loại,
còn generator phát ở **21 offset** của `art_table`. Dòng n−3 có tỉ lệ hiện (0.0986 / 0.1462) **cao hơn cả tỉ lệ phát**
(0.045), chỉ có thể do nguồn khác rơi vào bin mà phép lọc không loại. n−1 và n−2 vẫn đọc được vì tỉ lệ phát của chúng
cao hơn hẳn tỉ lệ hiện.

### Mục 0 (sửa phép lọc) và mục 2 — sổ kế toán sau khi sửa

Phép lọc "sạch một-nguồn" giờ loại **đủ 21 offset** của `art_table`, không chỉ 5 offset của bộ phân loại. Nó đổi
rất ít, nên chỗ rò không phải ở đó: chất gây nhiễu thật là **nhiễu nền**, phát ~0.07 mỗi bin trống.

| offset | n cha > 2k | rate dùng | phát | sống / đã phát | cao TB artefact | twin hiện | REAL hiện |
|---|---|---|---|---|---|---|---|
| n−1 | 4683 | 0.7987 | 0.8001 | **0.9653** | 289.8 | 0.7912 | **0.8754** |
| n+1 | 5076 | 0.4824 | 0.4868 | **0.7094** | 52.6 | 0.4052 | **0.4503** |
| n−2 | 4923 | 0.5509 | 0.5505 | **0.4849** | 20.6 | 0.3250 | **0.4050** |
| n+2 | 6518 | 0.0452 | 0.0453 | 0.3119 | 11.8 | 0.0824 | 0.0834 |
| n−3 | 5384 | 0.0447 | 0.0418 | 0.4133 | 15.3 | 0.0981 | 0.1480 |

**n+2 và n−3 KHÔNG đọc được bằng thước này**: tỉ lệ hiện (0.08–0.15) cao hơn cả tỉ lệ phát (0.042–0.045), nên phần
lớn cái nhìn thấy ở đó là nhiễu nền, không phải artefact. n+2 twin 0.0824 so với real 0.0834 gần bằng nhau chính vì
cả hai đều là nhiễu — mà nhiễu vừa được sửa ở mục 1. Các lớp yếu cần một thước khác.

**Ba offset mạnh đọc được**, trừ ~0.06 phần nhiễu:

| offset | cần gì để đạt real | đang có | kết luận |
|---|---|---|---|
| n−1 | tỉ lệ phát ≥ 0.82 | rate 0.800, survival 0.965 | **rate là ràng buộc chặn** — survival dù đạt 1.0 cũng không tới |
| n+1 | survival ≈ 0.80 | 0.709 | **survival thiếu** |
| n−2 | survival ≈ 0.63 | 0.485 | **survival thiếu** |

Nên mục 2′ **tách thành hai việc khác nhau**: n−1 là **rate**, n+1 và n−2 là **survival** (và `ART_SURV_P` đúng là
đường của riêng n−1 đang áp cho mọi offset, như `calibrate.py` tự ghi).

### Mục 2′b-2 (survival theo offset) — BỊ BÁC

Đo survival theo **chiều cao thực hiện của chính artefact**, không theo trung bình:

| chiều cao | n−1 | n+1 | n−2 | n+2 | n−3 | bảng × CARR |
|---|---|---|---|---|---|---|
| <8 | — | 0.152 | 0.135 | 0.109 | 0.125 | **0.154** |
| 8–12 | — | 0.291 | 0.290 | 0.318 | — | **0.304** |
| 12–16 | — | 0.457 | 0.450 | — | — | **0.448** |
| 16–21 | — | 0.623 | 0.587 | — | — | **0.569** |
| 21–32 | — | 0.748 | 0.765 | — | — | **0.730** |
| 32–51 | 0.901 | 0.894 | 0.877 | — | — | 0.853 |
| 51–85 | 0.925 | 0.952 | 0.908 | — | — | 0.905 |
| 85–148 | 0.952 | 0.972 | — | — | — | 0.915 |
| >148 | 0.979 | 0.978 | — | — | — | 0.924 |

**Mọi offset nằm trên MỘT đường, và khớp bảng.** "n−2 sống 0.485" chỉ vì artefact của nó toàn nằm ở vùng thấp (trung
bình 20.6 RFU) nơi survival vốn 0.5–0.6 — lại là lỗi so sánh ở giá trị trung bình trên phân bố lệch, đúng loại đã
gặp ở Cấp 9 với sd. `ART_SURV_P` là đường của n−1 nhưng **áp cho mọi offset lại đúng**, vì survival là thuộc tính của
chiều cao đỉnh chứ không của offset.

**Và điều đó khoá lại đáp án:** rate đúng luật, survival đúng luật ⇒ phần hụt phải nằm ở **tỉ lệ phát**. Với n−1 đó là
chặn cứng: real hiện 0.875 ở cha sáng mà rate chỉ 0.800.

**Cách đo `rate` sạch, từ chính bảng này:** ở cha sáng artefact cao nên survival ≈ 1, nên **tỉ lệ hiện của real là ước
lượng trực tiếp của tỉ lệ phát** (sau khi trừ nền nhiễu ~0.06):

| offset | real hiện ở cha > 2k | rate đang dùng | cần |
|---|---|---|---|
| n−1 | 0.875 | 0.800 | **+0.08** |
| n+1 | 0.450 | 0.482 | ~đủ |
| n−2 | 0.405 | 0.551 | ~đủ |

Nên **chỉ n−1 thiếu rate**; n+1 và n−2 có rate đủ, và phần hụt nhìn-thấy của chúng là do artefact **quá mờ** (phân bố
chiều cao phát ra nằm thấp), tức thuộc luật tỉ lệ chiều cao cho **toàn bộ** artefact phát ra — mà phép đo tỉ lệ trước
đây chỉ thấy được những cái ĐÃ nhìn thấy, tức đã bị chọn lọc lên trên.

### Mục 2′b-1 (rate của n−1) — cài, đo, và RÚT

Dẫn `rate` theo locus từ **cha sáng**, nơi survival ≈ 0.96–0.98 nên khả năng nhìn thấy ≈ khả năng phát; trừ nền nhiễu
rồi chia survival, cả hai lấy từ luật đã cài chứ không từ twin. `calibrate_art_rate.py` (mới) cho: cần **0.8621**,
bảng cũ **0.8265**, và 11/22 locus thiếu > 0.06 — ở mười locus real cho n−1 ở **mọi** cha sáng.

Cài vào và đo:

| | trước | sau |
|---|---|---|
| rate ở cha sáng | 0.796 | **0.849** |
| n−1 hiện ở cha sáng | 0.781 | **0.835** (real 0.875); khoảng cách 0.095 → **0.040** |
| NOC1 n−1 đỉnh / RFU | −0.43 / −3.4 % | **+0.33 / +3.2 %** |
| hỗn hợp n−1 đỉnh | −0.51 / +0.08 / +0.11 / +0.35 | **+0.15 / +1.01 / +1.26 / +1.02** |
| hỗn hợp n−1 RFU | −1 / +5 / +4 / −9 % | **+2.5 / +11.8 / +10.0 / −2.7 %** |
| tổng đỉnh | −0.78 / +2.06 / +2.27 / +2.93 | −1.24 / +2.81 / +2.43 / +4.18 |

Nó làm đúng việc được thiết kế nhưng **bảng từng-con-số xấu đi**: trên NOC1 chỉ **đổi dấu** lỗi chứ không giảm độ lớn,
và trên hỗn hợp n−1 vượt hẳn. Và đúng nguyên tắc "không có luật riêng của hỗn hợp": +0.33 đỉnh ở NOC1 nhân với 2.7 lần
số allele đang đứng cho +0.9, khớp +1.0 đo được — **một lỗi được nhân lên**, không phải lỗi mới.

**Phần dư 0.040 ở cha sáng có lẽ không phải stutter.** Ghi chú dự án: real có 6.7–8.5 % đỉnh bất thường trong ngữ cảnh
**pull-up** so với twin 0.9–1.8 %; phép lọc chỉ loại láng giềng ≥ 1000 RFU nên pull-up từ láng giềng thấp hơn vẫn lọt.
Nâng `rate` của stutter để phủ nó là fit một luật sai.

**Đã rút**: `make_insilico.py` về cấu hình sau mục 1; `data/art_rate.json` chuyển ra scratchpad;
`code/calibrate_art_rate.py` giữ lại kèm ghi chú đầy đủ rằng nó CHƯA được nối và vì sao. Phép **căn giữa `lrate`**
cũng bị bỏ theo, và đúng nên bỏ: nó không đổi gì đo được (0.7987 → 0.7960).
