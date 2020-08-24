import sys
sys.path.append('/home/wollex/Data/Science/PhD/Programs/ROI-matching/')
from Sheintuch_matching_prob import *


SH = Sheintuch_matching('/media/wollex/Analyze_AS1/linstop','762',range(1,113),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()
SH = Sheintuch_matching('/media/wollex/Analyze_AS1/linstop','231',range(1,88),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()
SH = Sheintuch_matching('/media/wollex/Analyze_AS1/linstop','232',range(1,75),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()
SH = Sheintuch_matching('/media/wollex/Analyze_AS1/linstop','236',range(1,29),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()

# SH = Sheintuch_matching('/media/wollex/Analyze_AS1/linstop','243',range(1,72),dataSet='redetect',s_corr_thr=0.2)
# SH.run_matching()
# SH = Sheintuch_matching('/media/wollex/Analyze_AS1/linstop','245',range(1,74),dataSet='redetect',s_corr_thr=0.2)
# SH.run_matching()
# SH = Sheintuch_matching('/media/wollex/Analyze_AS1/linstop','246',range(1,64),dataSet='redetect',s_corr_thr=0.2)
# SH.run_matching()

SH = Sheintuch_matching('/media/wollex/Analyze_AS1/Shank','918shKO',range(1,29),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()
SH = Sheintuch_matching('/media/wollex/Analyze_AS1/Shank','931wt',range(1,28),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()
SH = Sheintuch_matching('/media/wollex/Analyze_AS1/Shank','943shKO',range(1,28),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()

SH = Sheintuch_matching('/media/wollex/Analyze_AS3/Data','756',range(1,29),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()
SH = Sheintuch_matching('/media/wollex/Analyze_AS3/Data','757',range(1,28),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()
SH = Sheintuch_matching('/media/wollex/Analyze_AS3/Data','758',range(1,25),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()

SH = Sheintuch_matching('/media/wollex/Analyze_AS3/Data','65',range(1,45),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()
SH = Sheintuch_matching('/media/wollex/Analyze_AS3/Data','66',range(1,46),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()
SH = Sheintuch_matching('/media/wollex/Analyze_AS3/Data','72',range(1,45),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()

SH = Sheintuch_matching('/media/wollex/Analyze_AS3/Data','879',range(1,16),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()
SH = Sheintuch_matching('/media/wollex/Analyze_AS3/Data','884',range(1,25),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()

SH = Sheintuch_matching('/media/wollex/Analyze_AS3/Data','34',range(1,23),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()
SH = Sheintuch_matching('/media/wollex/Analyze_AS3/Data','35',range(1,23),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()

SH = Sheintuch_matching('/media/wollex/Analyze_AS1/others','549',range(1,29),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()
SH = Sheintuch_matching('/media/wollex/Analyze_AS1/others','551',range(1,28),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()

SH = Sheintuch_matching('/media/wollex/Analyze_AS3/Data','839',range(1,25),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()
SH = Sheintuch_matching('/media/wollex/Analyze_AS3/Data','840',range(1,25),dataSet='redetect',s_corr_thr=0.2)
SH.run_matching()
