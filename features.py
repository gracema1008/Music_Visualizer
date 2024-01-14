import os,sys
import numpy as np
import librosa


def map_note_to_rgb(val_arr,mood='happy',is_midi=True):
    '''
    颜色和情感：
    欢快- 橙色到绿色 start_color = [237, 175, 145] end_color = [145, 237, 175]
    冷静- 蓝绿色到紫色 start_color = [107, 146, 181] end_color = [146, 107, 181]
    激昂- 红色到黄色 start_color = [232, 79, 79] end_color = [232, 232, 79]
    伤感- 蓝色到蓝灰色 start_color = [76, 105, 173] end_color = [76, 81, 99]
    甜美- 粉色到浅粉色 start_color = [173, 104, 162] end_color = [245, 196, 237]
    '''
    # print("颜色和情感：\n欢快- 橙色到绿色\n冷静- 蓝绿色到紫色\n激昂- 红色到黄色\n伤感- 蓝色到蓝灰色\n甜美- 粉色到浅粉色")
    # mood = input("请输入曲子的情感：")
    if mood == 'happy' or mood== "欢快":
        start_color = np.array([237, 175, 145])
        end_color = np.array([145, 237, 175])
    if mood=='calm' or mood == "冷静":
        start_color = np.array([107, 146, 181])
        end_color = np.array([146, 107, 181])
    if mood =='excited' or mood== "激昂":
        start_color = np.array([232, 79, 79])
        end_color = np.array([232, 232, 79])
    if mood == 'sad' or mood=="伤感":
        start_color = np.array([76, 105, 173])
        end_color = np.array([76, 81, 99])
    if mood =='sweet' or mood==  "甜美":
        start_color = np.array([173, 104, 162])
        end_color = np.array([245, 196, 237])

    # 音高映射到RGB元组
    num_vals = len(val_arr)
    colors = []
    for i in range(num_vals):
        c = start_color + (i / (num_vals - 1)) * (end_color - start_color)
        colors.append(c)
    idx_clr=0
    rgb_colors={}
    if is_midi:# val to note and rgb
        for val in val_arr:
            note=librosa.midi_to_note(val)
            rgb_colors[note]= tuple(colors[idx_clr])
            idx_clr+=1
    else: # beat to color
        for val in val_arr:
            rgb_colors[val]= tuple(colors[idx_clr])
            idx_clr+=1
    return rgb_colors

def moving_average(interval, windowsize=50):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

def f0_to_note_and_rgb(f0s,mood='happy'):
    N = len(f0s)
    if np.isnan(f0s[0]) or f0s[0]==0:
        for k in range(1,N):
            if np.isnan(f0s[k])==False and f0s[k]>0:
                break
        if k==N-1: #全0 或只有最后一个有值，均视为非法
            print('input f0s is all illegal')
            return None
        f0s[0]=f0s[k]

    f0s=moving_average(f0s)

    for idx in range(1,len(f0s)): #从1开始
        if f0s[idx]==0 or np.isnan(f0s[idx]):
            f0s[idx]=f0s[idx-1]
    fmin=np.min(f0s)
    fmax=np.max(f0s)
    v_min= np.floor(librosa.hz_to_midi(fmin)) #MIDI音高值,A4对应MIDI音高值为69
    v_max= np.ceil(librosa.hz_to_midi(fmax))
    # 有半音！
    val_arr = np.arange(v_min,v_max+0.5,0.5)
    rgb_colors=map_note_to_rgb(val_arr,mood)

    rgbs=[]
    notes = []         #
    for f0 in f0s:
        note =  librosa.hz_to_note(f0)
        # print('note={}'.format(note))
        rgb=rgb_colors[note] #KeyError: 'A5'
        rgbs.append(rgb)
        notes.append(note) #for 显示
    return [notes,rgbs,rgb_colors]

def beats_to_rgb(idx_beats,mood='excited'):
    # N = len(idx_beats)
    v_min=min(idx_beats)
    v_max = max(idx_beats)
    val_arr = np.arange(v_min,v_max+1)
    rgb_colors=map_note_to_rgb(val_arr,mood,is_midi=False)
    beats_rgbs=[]
    for idx_b in idx_beats:
        rgb=rgb_colors[idx_b]
        beats_rgbs.append(rgb)
    return [beats_rgbs,rgb_colors]


def beat_by_mm(fnwav):
    print('calc beats ...')
    from madmom.features import (DBNDownBeatTrackingProcessor, RNNDownBeatProcessor)
    rnn = RNNDownBeatProcessor()
    act=rnn(fnwav)
    dbn = DBNDownBeatTrackingProcessor(beats_per_bar=[2,3,4,6], fps=100)
    beats = dbn(act)
    return beats

class song_features():
    def __init__(self,fmin=20,fmax=2000):
        self.freq_min=fmin
        self.freq_max=fmax 
        self.fnwav=None

    def get_f0s(self,y,fs):
        print('calc f0s ...')
        # y, fs = librosa.load(fnwav,mono=True)
        frmlen=2048
        frmhop=frmlen//4
        f0, voiced_flag, voiced_probs = librosa.pyin(y,fmin=self.freq_min,fmax=self.freq_max,
                                                     frame_length=frmlen,hop_length=frmhop)

        # t = librosa.times_like(f0) #间隔默认0.023s,(i.e. 512smps)
        lx = len(y)
        t = np.arange(0,lx+frmlen,frmhop) / fs
        lf = len(f0)
        t = t[:lf]
        sub=np.where(np.isnan(f0)) # nan下标
        f0[sub]=0
        return [t,f0]

    def get_power(self,y,fs):  # 计算功率谱
        print('calc specs ...')
        fw = fs/2
        frmhop=256 # 16kHz，=16ms
        frmlen=512
        # 默认参数： hop_length=512 n_fft=2048
        power = librosa.feature.melspectrogram(y=y, sr=fs,
                                               n_fft=frmlen,hop_length=frmhop,win_length=frmlen,
                                               n_mels=128, fmax=fw)
        percep_power = librosa.power_to_db(power, ref=np.max)
        # percep_power=np.power(power+1e-7,0.5)
        vmin = np.min(percep_power)

        normd_power = (percep_power-vmin) # >=0
        normd_power =normd_power/np.max(normd_power) # [0,1]
        melbins,nfrms=power.shape
        frmrate=frmhop/fs
        t = np.arange(0,nfrms)*frmrate
        return [t,normd_power]

    def extract_all_features(self,fnwav,fnfeature=None):
        if fnfeature!=None and os.path.exists(fnfeature) :
            print('get feature from npy file')
            fts_dict = np.load(fnfeature,allow_pickle=True).item()
        else:
            print('calc features..., which may take a long time')
            if False:
                y, fs = librosa.load(fnwav,mono=True,sr=16000) #sr降低计算量以及存储的文件大小
                # for歌曲：little_star.mp3
                # 报错： y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)
            else:
                fnout='%s_mono_16kHz.wav'%(fnwav[:-4])
                strcmd='ffmpeg -y -i %s -ac 1 -ar 16000 %s'%(fnwav,fnout)
                os.system(strcmd)
                y, fs = librosa.load(fnout,mono=True,sr=16000)
            if False:
                bpm, beats = librosa.beat.beat_track(y=y, sr=fs)
                beat_times = librosa.frames_to_time(beats, sr=fs)
                onset_times = librosa.onset.onset_detect(y=y, sr=fs,units='time')
            else:
                beats=beat_by_mm(fnwav)
                t_beats=beats[:,0]*1000 # in ms
                idx_beats=beats[:,1].astype(np.int)
            tp,power = self.get_power(y,fs) # mel_binss*nfrms
            tf,f0s=self.get_f0s(y,fs)
            t_pwrs = (tp*1000)
            t_f0s = (tf*1000) # s to ms
            # t_beats = (beat_times*1000)
            # f0s_notes,f0s_rgbs=self.f0_to_note_and_rgb(f0s)

            fts_dict={
                "wav":[y,fs],
                "power":[t_pwrs,power],
                "f0s":[t_f0s,f0s],
                'beats':[t_beats,idx_beats]}
            if fnfeature!=None:
                np.save(fnfeature,fts_dict)

        return fts_dict




def test_features(datadir,ftitle):
    fnwav='%s/%s.mp3'%(datadir,ftitle)
    if os.path.exists(fnwav)==False:
        fnwav='%s/%s.wav'%(datadir,ftitle)
    if os.path.exists(fnwav)==False:
        print('mp3 or wav file not exists: %s'%(fnwav))
        return None
    fnfeature='%s/%s.npy'%(datadir,ftitle)

    hdl_fts = song_features()
    fts_dict = hdl_fts.extract_all_features(fnwav,fnfeature)
    return fts_dict

def jingle_bells():
    datadir='../data'
    ftitles=['jingle_bells','little_star','symphony_5']
    ftitle=ftitles[0]
    fts_dict=test_features(datadir,ftitle)
    y,fs = fts_dict['wav']
    t_pwrs,power = fts_dict['power']
    t_f0s,f0s = fts_dict['f0s']
    t_beats,idx_beats = fts_dict['beats']




    nf=len(f0s)
    if True:
        ftitle2='jingle_4'
        fts_dict2 =test_features(datadir,ftitle2)
        t_f0s2,f0s2 = fts_dict2['f0s']
        if fts_dict==None:
            print('error')
            return

        tms_s=  11.274*1000
        tms_e = 61.301*1000
        for k in range(1,nf):
            tcur=t_f0s[k]
            if tcur>=tms_s and tcur<=tms_e:
                f0s[k] = f0s2[k]
    if True:
        fts_dict={
            "wav":[y,fs],
            "power":[t_pwrs,power],
            "f0s":[t_f0s,f0s],
            'beats':[t_beats,idx_beats]}
        fnfeature2='%s/%s.npy'%(datadir,ftitle)
        np.save(fnfeature2,fts_dict)





    fnf0s='%s/%s_f0s_modif.txt'%(datadir,ftitle)

    with open(fnf0s,'w') as fw:
        for k in range(1,nf):
            ts = t_f0s[k-1]/1000
            te=t_f0s[k]/1000
            str_f0s = '%f %f %f'%(ts,te,f0s[k])
            fw.write('%s\n'%(str_f0s))
    print('f0s num =%d'%nf)


    fnbeats ='%s/%s_beats.txt'%(datadir,ftitle)
    nb = len(t_beats)
    with open(fnbeats,'w') as fw:
        for k in range(nb):
            str_beats = '%f %d'%(t_beats[k]/1000,idx_beats[k])
            fw.write('%s\n'%(str_beats))
    print('beats num = %d'%nb)


if __name__=='__main__':
    datadir='../data'
    ftitles=['jingle_bells','little_star','symphony_5']
    # jingle_bells()

    for ftitle in ftitles:
        print('fitlle is %s'%ftitle)
        fts_dict=test_features(datadir,ftitle)
        if True:
            fncrepe='%s/%s_2.crepe.f0s'%(datadir,ftitle)
            with open(fncrepe) as fr:
                lines = fr.readlines()
            tarr= []
            f0arr=[]
            for line in lines:
                arr = line.strip().split()
                if len(arr)!=2:
                    continue
                tarr.append(float(arr[0])*1000)
                f0arr.append(float(arr[1]))
            tarr =np.array(tarr)
            f0arr = np.array(f0arr)
            print('%s: crepe f0s: %d'%(ftitle,len(f0arr)))
            fts_dict["f0s"] = [tarr,f0arr]
            fnfeature2='%s/%s_crepe_modif.npy'%(datadir,ftitle)
            np.save(fnfeature2,fts_dict)


        # save diff features in txt
        if False:
            y,fs = fts_dict['wav']
            t_pwrs,power = fts_dict['power']
            t_f0s,f0s = fts_dict['f0s']
            t_beats,idx_beats = fts_dict['beats']



            fnf0s='%s/%s_f0s.txt'%(datadir,ftitle)
            nf=len(f0s)
            with open(fnf0s,'w') as fw:
                for k in range(1,nf):
                    ts = t_f0s[k-1]/1000
                    te=t_f0s[k]/1000
                    str_f0s = '%f %f %f'%(ts,te,f0s[k])
                    fw.write('%s\n'%(str_f0s))
            print('f0s num =%d'%nf)


            fnbeats ='%s/%s_beats.txt'%(datadir,ftitle)
            nb = len(t_beats)
            with open(fnbeats,'w') as fw:
                for k in range(nb):
                    str_beats = '%f %d'%(t_beats[k]/1000,idx_beats[k])
                    fw.write('%s\n'%(str_beats))
            print('beats num = %d'%nb)

















