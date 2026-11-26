"""Close this fixed-scope transfer experiment and render its scientific figures."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
import run as job

HERE=Path(__file__).resolve().parent
LABELS={'roundabout':'RoundaboutHD — vòng xuyến','svmot':'SVMOT — che khuất',
        'helmet_c25':'HELMET — xe máy, c25','helmet_c10':'HELMET — xe máy, c10'}
METHODS={'kf':'KF','ekf':'EKF','ukf':'UKF','parent':'PF100 fixed R (R132)',
         'candidate':'PF100 R(confidence) (R142)'}


def figures(summary,events):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#9aa3af','#9aa3af','#9aa3af','#536173','#176ab4']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axes=plt.subplots(2,4,figsize=(17,9),layout='constrained')
    xs=np.arange(5);ticks=['KF','EKF','UKF','PF R132','PF R142']
    for col,(name,data) in enumerate(summary.items()):
        for row,(field,scale,ylabel) in enumerate((('HOTA',100,'HOTA (%) — cao hơn tốt hơn'),
                                                 ('IDSW',1,'IDSW (lần) — thấp hơn tốt hơn'))):
            ax=axes[row,col];values=[data['scores'][m]['mean'][field]*scale for m in job.MODES]
            ax.bar(xs,values,color=colors,width=.68)
            for x,m,value in zip(xs,job.MODES,values):
                seeds=[v[field]*scale for v in data['scores'][m]['seeds']]
                if len(seeds)>1:ax.errorbar(x,value,yerr=[[max(0,value-min(seeds))],[max(0,max(seeds)-value)]],
                                          fmt='none',ecolor='#17212f',capsize=3)
                ax.annotate(f'{value:.2f}' if field=='HOTA' else f'{value:.1f}',(x,value),
                            xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
            ax.set_xticks(xs,ticks,rotation=30,ha='right');ax.set_xlabel('Cấu hình bộ lọc')
            ax.set_ylabel(ylabel);ax.set_ylim(0,max(values)*1.25+1)
            ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
            if row==0:
                cfg=job.cases()[name]
                ax.set_title(LABELS[name]+f"\n{data['videos']} chuỗi · {data['frames']:,} ảnh"
                             +f"\n{cfg['detector']} · conf > {cfg['conf']}",fontsize=10)
    fig.suptitle('Kiểm chuyển PF mới trên các tập kịch bản — toàn phạm vi đã khóa',fontsize=15)
    fig.supxlabel('Nguồn: R143, 19/11/2026. PF: mean 3 seed; thanh lỗi là min–max seed, không phải CI.\n'
                  'Mỗi ô dùng thang riêng. Cùng detection trong từng điều kiện; KF/EKF/UKF/R132 fixed R, R142 có thêm R theo confidence. Cả hai PF dùng100 hạt.',fontsize=10)
    fig.savefig(HERE/'scenario_overview.png',dpi=180);plt.close(fig)

    choices=[('roundabout','turn25','Roundabout: đổi hướng ≥25°'),
             ('roundabout','turn35','Roundabout: đổi hướng ≥35°'),
             ('svmot','visibility_le_0_5','SVMOT: visibility ≤0,5')]
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for col,(case,group,title) in enumerate(choices):
        values_by_mode=events['means'][case]
        for row,field in enumerate(('IDSW','FN')):
            ax=axes[row,col];values=[values_by_mode[m][group][field] for m in job.MODES]
            ax.bar(xs,values,color=colors,width=.68)
            for x,value in zip(xs,values):ax.annotate(f'{value:.1f}',(x,value),xytext=(0,6),
                                                     textcoords='offset points',ha='center')
            ax.set_xticks(xs,ticks,rotation=25,ha='right');ax.set_xlabel('Cấu hình bộ lọc')
            ax.set_ylabel('IDSW (lần)' if field=='IDSW' else 'FN (hộp bị bỏ sót)')
            ax.set_ylim(0,max(values)*1.22+1);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            if row==0:ax.set_title(title+f"\n{values_by_mode['kf'][group]['GT_boxes']:,.0f} hộp GT")
    fig.suptitle('Sự kiện tracking trong các nhóm kịch bản — thấp hơn tốt hơn',fontsize=15)
    fig.supxlabel('Nguồn: R143, CLEAR ghép trên toàn chuỗi trước khi đếm trong mask. PF là trung bình 3 seed.\n'
                  'Góc thay đổi là góc trên ảnh; không reset tracker tại đoạn rẽ. Không phải HOTA của tập con.',fontsize=10)
    fig.savefig(HERE/'scenario_groups.png',dpi=180);plt.close(fig)


def main():
    frozen=job.frozen_check();proof=job.read(HERE/'verification.json')
    assert proof['status']=='PASS' and proof['units']==765
    analysis=job.read(HERE/'analysis_verification.json');assert analysis['status']=='PASS'
    assert analysis['whole_sequence_CLEAR_checks']==117
    for cert in (proof,analysis,job.read(HERE/'analysis_inputs.json')):
        for p,h in cert['sha256'].items():assert job.sha(p)==h,p
    summary=job.read(HERE/'summary.json');events=job.read(HERE/'scenario_events.json')
    helmet=job.read(HERE/'helmet_bootstrap.json')
    data={'status':'PASS','quality_units':765,'groups':36,'summary':summary,'scenario_events':events,
          'helmet_bootstrap':helmet,'source_hashes':len(frozen['sha256']),
          'scope':'Frozen R142 transfer, fixed-R controls; previously observed datasets.',
          'Nano_measured':False,'default_changed':False,'timing_benchmark':False}
    job.write(HERE/'final_summary.json',data)
    figures(summary,events)

    # Canonical state precedes the human report, as required by AGENTS.md.
    plan=job.ROOT/'handoff/PLAN_DA_CHOT.md';text=plan.read_text(encoding='utf-8')
    start=text.index('**R143 —');end=text.index('**R142 —',start)
    lines=['**R143 — kiểm chuyển PF R142 trên các tập kịch bản, hoàn tất765 lượt; kết quả mô tả theo từng tập.** '
           'Giữ nguyên bộ lọc/detection/GT/cấu hình R142, mới chạy KF/EKF/UKF fixed-R, PF100 R132 fixed-R và PF100 R142 confidence-R×3 seed.']
    for name,record in summary.items():
        scores=record['scores']
        ids='/'.join(f"{scores[m]['mean']['IDSW']:.2f}" for m in job.MODES)
        hota='/'.join(f"{100*scores[m]['mean']['HOTA']:.4f}" for m in job.MODES)
        lines.append(f"{LABELS[name]}: {record['videos']} chuỗi/{record['frames']} ảnh; IDSW KF/EKF/UKF/R132/R142 **{ids}**; HOTA% **{hota}**. Cổng mô tả {json.dumps(record['checks_descriptive'],ensure_ascii=False)}.")
    turn=events['means']['roundabout'];occ=events['means']['svmot']
    lines.append('Riêng nhóm rẽ ảnh>=25° (6601hộp), IDSW KF/EKF/UKF/R142 **'
                 +'/'.join(f"{turn[m]['turn25']['IDSW']:.2f}" for m in ('kf','ekf','ukf','candidate'))
                 +'**; nhóm>=35° (4225hộp) **'
                 +'/'.join(f"{turn[m]['turn35']['IDSW']:.2f}" for m in ('kf','ekf','ukf','candidate'))
                 +'**. Nhóm SVMOTvisibility<=0,5 (12051hộp), IDSW KF/EKF/UKF/R142 **'
                 +'/'.join(f"{occ[m]['visibility_le_0_5']['IDSW']:.2f}" for m in ('kf','ekf','ukf','candidate'))
                 +'**. PF mới tốt hơn về điểm IDSW ở nhóm rẽ, chưa tốt hơn KF/EKF/UKF ở nhóm che; '
                 'không khái quát từ toàn tập. HELMET CIIDSW soEKF/UKF còn chứa hoặc chạm0; '
                 'PF R132 có IDSW thấp hơn R142 ở cả hai ngưỡng dù HOTA thấp hơn. SVMOT R142 giảm0,642điểm HOTA soR132, không chọn mặc định toàn miền.')
    lines.append('Whole-sequence CLEAR khớp117 lượt trước phân tầng R99turn25/35/straight8 và SVMOTvisibility<=0,5; HELMETbootstrap5000 theo12nhóm cảnh mỗi điều kiện. '
                 'Raw/COMBINED/historicaldeterministic/source/cache/GT/MOT/CLS/timing/765marker đạt. '
                 '`logs/review143_pf_scenarios/{final_summary.json,final_verification.json,10_ket_qua.md,scenario_overview.png,scenario_groups.png}`. '
                 'Không suy CI độc lập từ4camera cùng nút giao hay9clip/2scene; không gọi test mù hoặc purePFsuperiority. '
                 'HELMET làtest36, Roundabout chỉprefix3000; không gọi full toàn bộ dataset. Không đổi src/default/Nano/khóa luận; không còn job chất lượng chạy.')
    plan.write_text(text[:start]+'\n\n'.join(lines)+'\n\n'+text[end:],encoding='utf-8')

    verdicts=[]
    for name,record in summary.items():
        scores=record['scores'];candidate=scores['candidate']['mean']
        verdict='ít IDSW hơn cả ba mốc' if record['checks_descriptive']['IDSW_below_all'] else 'chưa ít IDSW hơn cả ba mốc'
        refs='/'.join(f"{scores[m]['mean']['IDSW']:.0f}" for m in ('kf','ekf','ukf'))
        extra=''
        id_diff=candidate['IDSW']-scores['parent']['mean']['IDSW']
        if id_diff>1e-9:
            extra+=f' Nhiều hơn PF R132 trung bình {id_diff:.2f} lượt đổi ID.'
        elif abs(id_diff)<=1e-9:
            extra+=' IDSW trung bình bằng cùng PF R132.'
        if not record['checks_descriptive']['HOTA_vs_parent']:
            diff=100*(candidate['HOTA']-scores['parent']['mean']['HOTA'])
            extra+=f' HOTA giảm {abs(diff):.4f} điểm phần trăm so cùng PF R132.'
        if not record['checks_descriptive']['FN_vs_parent']:
            extra+=' FN tăng quá0,2% so cùng PF R132.'
        verdicts.append(f"\n- **{LABELS[name]}:** PF {candidate['IDSW']:.2f} IDSW so KF/EKF/UKF {refs}; {verdict}. "
                        f"HOTA PF {candidate['HOTA']*100:.4f}%."+extra)
    report=['# R143 — kết quả PF mới trên ba tập kịch bản',
            '\nHoàn tất ngày 19/11/2026: 49 chuỗi khác nhau, bốn điều kiện (HELMET có hai ngưỡng), đủ 765 lượt chạy/36 nhóm. '
            'Cùng detection giữa các bộ lọc trong mỗi điều kiện. '
            'PF chạy ba seed 0/1/2. Không chỉnh lại R142 theo điểm mới. Các bộ lọc deterministic được chạy mới và raw từng video khớp lịch sử.',
            '\nPF mới là Native Cartesian RBPF100 R142: chỉ lấy mẫu vận tốc góc, các trạng thái còn lại cập nhật Gaussian có điều kiện; '
            'chuyển score detection đang ghép vào `R(c)=max(1-c,0.05)R0`. PF đối chứng là cùng lõi R132, không dùng score để đổi R. '
            'Cột PF R132 ở đây không phải PF cổ điển500 hạt trong slide cũ.',
            '\n## Kết quả chính',*verdicts,'\n![Điểm toàn tập](scenario_overview.png)',
            '\n## Điểm toàn phạm vi đã khóa']
    for name,record in summary.items():
        report += ['\n### '+LABELS[name],
                   f"\n{record['videos']} chuỗi, {record['frames']:,} ảnh mỗi lượt. Cấu hình nguồn: `{job.cases()[name]['cache']}`; conf>{job.cases()[name]['conf']}." ]
        for m in job.MODES:
            r=record['scores'][m]['mean']
            report.append(f"\n- **{METHODS[m]}:** IDSW **{r['IDSW']:,.2f}**, HOTA **{r['HOTA']*100:.4f}%**, IDF1 {r['IDF1']*100:.4f}%, FN {r['FN']:,.2f}, FP {r['FP']:,.2f}.")
        seeds=[r['IDSW'] for r in record['scores']['candidate']['seeds']]
        report.append(f"\nIDSW R142 từng seed: {seeds}. Kiểm mô tả: `{record['checks_descriptive']}`. "
                      'Không biến kiểm mô tả thành lý do loại bỏ điều kiện hoặc sửa tham số.')
    report += ['\n## Nhóm rẽ và nhóm che khuất',
               '\nCLEAR được ghép trên toàn chuỗi; xác nhận tổng IDSW/FN khớp117 lượt trước khi đếm sự kiện trong nhóm. '
               'Không reset tracker ở đầu đoạn rẽ; không coi HOTA toàn tập là HOTA của riêng nhóm rẽ.']
    for case,group,title in [('roundabout','turn25','Đổi hướng ảnh≥25°'),('roundabout','turn35','Đổi hướng ảnh≥35°'),
                             ('svmot','visibility_le_0_5','Độ nhìn thấy native≤0,5')]:
        values=events['means'][case]
        report.append(f"\n**{title}:** {values['kf'][group]['GT_boxes']:,.0f} hộp GT.")
        for m in job.MODES:
            r=values[m][group]
            report.append(f"\n- {METHODS[m]}: IDSW **{r['IDSW']:.2f}**, FN **{r['FN']:.2f}**.")
    report += ['\n![Nhóm kịch bản](scenario_groups.png)','\n## Độ biến thiên HELMET']
    for case,record in helmet.items():
        report.append('\n**'+LABELS[case]+'**, bootstrap5.000 mẫu theo12 nhóm cảnh, lấy lại cả3 clip của nhóm:')
        for mode in ('kf','ekf','ukf','parent'):
            ids=record['candidate_minus_reference'][mode]['IDSW'];h=record['candidate_minus_reference'][mode]['HOTA']
            report.append(f"\n- R142−{METHODS[mode]}: IDSW {ids['difference']:.2f}, CI95% {ids['ci95']}; "
                          f"HOTA {h['difference']*100:+.4f} điểm phần trăm, CI95% {[100*v for v in h['ci95']]}.")
    report += ['\nCI IDSW so EKF/UKF trên HELMET vẫn chứa hoặc chạm0, nên ưu thế theo trung bình chưa đủ để khẳng định khác biệt chắc chắn. '
       'PF R132 có ít IDSW hơn R142 ở cả hai ngưỡng, dù R142 có HOTA cao hơn. Không chọn R142 làm mặc định toàn miền chỉ từ lượt này.',
       '\n## Giới hạn và kiểm chứng',
       '\n- R142 là PF kết hợp R theo confidence; controls KF/EKF/UKF vẫn fixed-R. Không suy ra PF thuần thắng khi mọi bộ lọc có cùng R(c). '
       'Confidence phân lớp chưa được hiệu chỉnh thành độ bất định vị trí.',
       '\n- Đây là chuyển cấu hình sang dữ liệu đã quan sát từ trước, không phải test mù. Roundabout4camera cùng nút giao và SVMOT9clip/2scene chỉ báo mô tả; '
       'không dùng CI theo video giả độc lập. CI HELMET vẫn có giới hạn theo cấu trúc nhóm và lịch sử chọn mô hình.',
       '\n- Roundabout chỉ frame1–3000 mỗi camera, SCT không gán nhãn xe đỗ nên dự đoán xe đỗ chịu FP. '
       'SVMOT dùng full-image của dự án, không ROI nguồn. Nhãn visibility không xác định loại vật che. '
       'HELMET là mẫu test36/12nhóm; box thường bao cả người ngồi, không có phân tầng rẽ/che riêng trong lượt này.',
       '\n- Tracking dùng cache đã khóa trên ổD; không sinh detection mới, không cần ảnh ổF đang ngắt. '
       'Cùng nguồncache/GT/hash và các thông số nhưgiao thức. Không đo tốc độ từ các job song song, chưa đo PF này trên Nano.',
       '\n- Preflight574hash, smoke đủ8 cặp đầu/cuối (mỗi cặp3historical controls,3route-off và3repeatPF). '
       'Lần preflight đầu dừng do tên trường GT legacy `gt_sha256` của SVMOT khác trường `gt_luc_sinh_sha256`; '
       'đã hỗ trợ đúng hai schema, vẫn yêu cầu hash chính xác, trước khi đóng inputs và chưa chạy full. Log đầu giữ nguyên.',
       '\n- COMBINED đối chiếu trực tiếp TrackEval; helper R66 giữ AssA=0 khi TP=0. '
       'Đã kiểm lại nguồn/marker/MOT/lớp/timing/raw. Chứng cứ: `inputs.json`, `smoke.json`, `verification.json`, '
       '`scenario_events.json`, `helmet_bootstrap.json`, `analysis_verification.json`, `final_summary.json`, `final_verification.json`.',
       '\nĐã cập nhật PLAN mục0 trước báo cáo. Không sửa src, mặc định, launcher Nano, slide hoặc khóa luận.']
    (HERE/'10_ket_qua.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    paths=[Path(__file__),HERE/'00_protocol.md',HERE/'inputs.json',HERE/'smoke.json',HERE/'verification.json',
           HERE/'summary.json',HERE/'analysis_inputs.json',HERE/'analysis_verification.json',HERE/'scenario_events.json',
           HERE/'helmet_bootstrap.json',HERE/'final_summary.json',HERE/'10_ket_qua.md',HERE/'scenario_overview.png',
           HERE/'scenario_groups.png',plan]
    job.write(HERE/'final_verification.json',{'status':'PASS','date_utc':datetime.now(timezone.utc).isoformat(),
          'quality_units':765,'groups':36,'scenario_CLEAR_parity_checks':117,
          'sha256':{str(p):job.sha(p) for p in paths}})
    print('FINAL PASS: 765 quality units, 117 scenario checks, both HELMET conditions retained',flush=True)


if __name__=='__main__':main()
