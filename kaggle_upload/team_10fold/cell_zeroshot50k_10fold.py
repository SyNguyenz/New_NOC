# ZERO-SHOT 50k --conditioned, 10 FOLD — MOI NGUOI CHAY 2 FOLD TRONG MOT NOTEBOOK (~7 gio)
# Huong dan day du: doc/chay_10fold_zeroshot50k.md. CHI SUA hai dong NGUOI / FOLDS ngay duoi.
#
# Moi fold: Phase 1 150 epoch tren 50.000 hon hop sinh + don nguon that cua fold -> dem NoC bang logits_card
# (zero-shot, KHONG mot nhan hon hop that nao, giai ma argmax chot truoc) -> ID top-k voi k do.
# RUN_BUDGET=True: them 3 lan fine-tune voi DUNG ngan sach nhan cua deepNoC (371 ho so: 34/87/79/93/78), tren chinh
# backbone vua train (--init_from, ~12 phut/fold). Moi lan rut chay BAY cach tren cung cac hang: card_cal, card_ft,
# head_only (doc diem ID), head_free / head_mac / head_unf (KHONG doc diem ID), lora_inv. Cach dung cho dong
# fine-tune chinh duoc chon tren fold 0 (doc chay_10fold_zeroshot50k.md, muc 1b).
# Backbone dong bang trong moi buoc; [ID GUARD] kiem ID khong doi.

NGUOI = 'ten_ban'          # vd 'manh', 'nguyen', 'dung', 'quang', 'giang'
FOLDS = [0, 1]             # 2 fold duoc phan cong trong doc/chay_10fold_zeroshot50k.md
RUN_BUDGET = True

EPOCHS = 150
CODE_HASH = '529535b846'
BUDGET = '34,87,79,93,78'
DRAWS = (0, 1, 2)
FT_ARMS = ('card_cal', 'card_ft', 'head_only', 'head_free', 'head_mac', 'head_unf', 'lora_inv')  # cai cuoi la --count
# md5(noc_train.npy) cua tung bo du lieu, sinh tren MOT may bang kaggle_upload/tools/build_cond50k_folds.py.
# Sai md5 = dataset khac ban da chot -> cell dung lai.
MANIFEST = {'0': '9ebc71ae976a495d803feebbbd788c5d',
            '1': '9159b97de660e3eb7d5970f382d1e6b3',
            '2': '601344dcc9e738f1858ca0523e22d1cc',
            '3': 'aea8c3b4d4de9d74e460aaaadbe0a77a',
            '4': 'da2e76a4c0893c9626b6afe5f80bc099',
            '5': '4d27f27b534a93bedeaae634e233aff4',
            '6': '3bc5ed6b4b55a1aef5f1832d4e247f5f',
            '7': '67d83fdf046102389930faf8837e2489',
            '8': '8a29df46418a1fd04be1754d72a53ae0',
            '9': '514225ba3e1d51698c97e9a589c24882'}

import glob, os, json, shutil, subprocess, hashlib, time, traceback
import numpy as np

assert NGUOI != 'ten_ban', 'sua NGUOI thanh ten ban'
src = glob.glob('/kaggle/input/**/kaggle_run_increment1.py', recursive=True)
assert len(src) == 1, f'can dung 1 dataset code (noc-fold-code), dang thay: {src}'
CODE = '/kaggle/working/noc'; shutil.copytree(os.path.dirname(src[0]), CODE, dirs_exist_ok=True)
h = hashlib.md5(b''.join(open(f'{CODE}/{f}', 'rb').read() for f in
                ('train_set_transformer.py', 'noc_lora.py', 'fold_report.py',
                 'kaggle_run_increment1.py', 'models/set_transformer.py'))).hexdigest()[:10]
assert h == CODE_HASH, f'dataset code la CODE {h}, can {CODE_HASH} -> Input -> Check for updates'
print(f'CODE {h} OK | nguoi {NGUOI} | fold {FOLDS} | budget {RUN_BUDGET}')

DATA = {}
for F in FOLDS:
    c = glob.glob(f'/kaggle/input/**/data_cond50k_fold{F}/fold_info.json', recursive=True)
    assert len(c) == 1, f'fold {F}: can dung 1 thu muc data_cond50k_fold{F}, dang thay {c} -> Add Data noc-cond50k-fold{F}'
    d = os.path.dirname(c[0])
    fi = json.load(open(f'{d}/fold_info.json')); bi = json.load(open(f'{d}/build_info.json'))
    m = hashlib.md5(open(f'{d}/noc_train.npy', 'rb').read()).hexdigest()
    assert fi['fold'] == F and bi['fold'] == F, f'fold {F}: thu muc khai fold {fi["fold"]}'
    assert m == bi['md5']['noc_train.npy'], f'fold {F}: noc_train.npy khong khop build_info.json'
    if MANIFEST:
        assert m == MANIFEST[str(F)], f'fold {F}: md5 {m} khac ban da chot {MANIFEST[str(F)]}'
    DATA[F] = d
    print(f'  fold {F}: {d} | unknown {fi["unknown_donors"]} | train NOC {bi["noc_train_counts"]} | md5 {m[:8]}')

SEND = '/kaggle/working/send'; os.makedirs(SEND, exist_ok=True)


def run(cmd, log, res=None, save_to=None, show=None):
    """Chay mot lenh, ghi log; neu save_to: chep backbone NGAY khi Phase 1 xong (da mat backbone ba lan)."""
    saved = save_to is None
    with open(log, 'w') as lf:
        p = subprocess.Popen(cmd, cwd=CODE, env=dict(os.environ, WORK_DIR='/kaggle/working', PYTHONUNBUFFERED='1'),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            lf.write(line); lf.flush()
            if show is None or any(k in line for k in show):
                print(line, end='', flush=True)
            if not saved and line.startswith('Training done in') and os.path.exists(f'{res}/best_model.pt'):
                os.makedirs(save_to, exist_ok=True)
                for nm in ('best_model.pt', 'best_model.fold.json'):
                    shutil.copy(f'{res}/{nm}', f'{save_to}/{nm}')
                saved = True
                print(f'>>> backbone da luu -> {save_to}', flush=True)
    return p.wait()


SUMMARY = {}
for F in FOLDS:
    t0 = time.time(); out = f'{SEND}/fold{F}'; os.makedirs(out, exist_ok=True)
    try:
        tag = f'zs50k_fold{F}'; res = f'{CODE}/results/{tag}_seed42'; bb = f'{out}/backbone'
        print(f"\n{'#' * 90}\n# FOLD {F}: Phase 1 + zero-shot\n{'#' * 90}", flush=True)
        cmd = ['python', 'kaggle_run_increment1.py', '--fold', str(F), '--data', DATA[F], '--epochs', str(EPOCHS),
               '--out_subdir', tag, '--protocol', 'strict', '--count', 'zero_shot', '--compare', 'gate',
               '--count_decode', 'argmax']
        rc = run(cmd, f'{out}/{tag}.log', res=res, save_to=bb)
        assert rc == 0, f'zero-shot fold {F} that bai (rc {rc}) — xem {out}/{tag}.log'
        shutil.copy(f'{res}/report.json', f'{out}/fold{F}_seed42.json')      # file gop bang report_folds.py
        shutil.copy(f'{res}/metrics.json', f'{out}/{tag}.metrics.json')
        M = json.load(open(f'{res}/metrics.json')); R = json.load(open(f'{res}/report.json'))
        a = M['count_arms']['zero_shot']
        SUMMARY[F] = dict(zs=a['macro_f1'], zs_alt=a.get('macro_f1_expect_decode'), f5=a['per_noc_f1'],
                          acc=a['accuracy'], open_mix=(a.get('open') or {}).get('mixtures_macro_f1'),
                          em=M['em_at_pred_k'], oracle=M['oracle_em'],
                          em_noreal=M['alpha_dev_no_real_label']['em_at_pred_k'], auroc=M['reject_auroc'],
                          fp=R['checks']['id_fp_logits_cls'],
                          ok=R['checks']['params_unchanged'] is True and R['checks']['split_audit_shared_combos'] == 0,
                          budget=[])
        if RUN_BUDGET:
            for dr in DRAWS:
                btag = f'budget_fold{F}_d{dr}'
                cmd = ['python', 'kaggle_run_increment1.py', '--fold', str(F), '--data', DATA[F], '--init_from', bb,
                       '--out_subdir', btag, '--protocol', 'both', '--count', 'lora_inv',
                       '--compare', *FT_ARMS[:-1], '--lora_epochs', '5',
                       '--lora_budget', BUDGET, '--lora_budget_seed', str(dr), '--skip-prep']
                print(f'\n# FOLD {F}: fine-tune 371 nhan, lan rut {dr}', flush=True)
                rc = run(cmd, f'{out}/{btag}.log',
                         show=('[ID GUARD]', '[DEEPNOC]', '[K-fold pooled]', 'Error', 'Traceback'))
                assert rc == 0, f'budget fold {F} draw {dr} that bai — xem {out}/{btag}.log'
                bres = f'{CODE}/results/{btag}_seed42'
                shutil.copy(f'{bres}/metrics.json', f'{out}/{btag}.metrics.json')
                bA = json.load(open(f'{bres}/metrics.json'))['count_arms']
                bR = json.load(open(f'{bres}/report.json'))
                SUMMARY[F]['budget'].append(dict(fp=bR['checks']['id_fp_logits_cls'], **{
                    a: dict(strict=bA[a]['macro_f1'], deepnoc=(bA[a].get('deepnoc') or {}).get('macro_f1'))
                    for a in FT_ARMS}))
        SUMMARY[F]['minutes'] = round((time.time() - t0) / 60)
    except Exception:
        traceback.print_exc()
        SUMMARY[F] = {'error': traceback.format_exc().splitlines()[-1]}
        print(f'!!! FOLD {F} LOI — van chay fold tiep theo. Gui log trong {out}/ cho Manh.', flush=True)
    json.dump(SUMMARY, open(f'{SEND}/summary_{NGUOI}.json', 'w'), indent=1)

print(f"\n{'=' * 100}\n===== KET QUA {NGUOI}: fold {FOLDS} =====\n{'=' * 100}")
for F, s in SUMMARY.items():
    if 'error' in s:
        print(f'fold {F}: LOI -> {s["error"]}'); continue
    print(f"fold {F}: zero-shot macroF1 {s['zs']:.4f} (expect {s['zs_alt']}) | acc {s['acc']:.4f} | F1 NOC1..5 "
          + ' '.join(f'{x:.3f}' for x in s['f5']) + f" | ngoai panel hon hop {s['open_mix']}")
    print(f"        ID: EM@k {s['em']} | oracle {s['oracle']} | EM@k alpha khong nhan {s['em_noreal']} | "
          f"reject AUROC {round(s['auroc'], 4)} | fp {s['fp']} | sach {s['ok']} | {s.get('minutes')} phut")
    if s['budget']:
        same = all(b['fp'] == s['fp'] for b in s['budget'])
        print(f"        371 nhan, 3 lan rut (ID khong doi {same}):")
        for a in FT_ARMS:
            st = [b[a]['strict'] for b in s['budget']]; dn = [b[a]['deepnoc'] for b in s['budget']]
            print(f"          {a:<10} strict {np.mean(st):.4f} +- {np.std(st, ddof=1):.4f} | "
                  f"deepnoc {np.mean(dn):.4f} +- {np.std(dn, ddof=1):.4f}")

shutil.make_archive(f'/kaggle/working/send_{NGUOI}', 'zip', SEND)
print(f'\nGUI CHO MANH: /kaggle/working/send_{NGUOI}.zip (tab Output cua version nay)')
