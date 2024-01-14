import pygame
import numpy as np
from features import *
from scipy.signal import savgol_filter
from random import randint

def show_note_rgbs(notes_rgbs_dict,rct,screen):
    rct_note_clr_x,rct_note_clr_y,rct_note_clr_w,rct_note_clr_h=rct
    N = len(notes_rgbs_dict)
    rct_note_clr_w=rct_note_clr_w/N
    for idx,note in enumerate(notes_rgbs_dict.keys()):
        color=notes_rgbs_dict[note]
        x =rct_note_clr_x+rct_note_clr_w*idx
        rect = pygame.Rect(x,rct_note_clr_y, rct_note_clr_w, rct_note_clr_h)
        pygame.draw.rect(screen, tuple(color), rect)
        pygame.draw.rect(screen, 'grey', rect,1)
        # ---------------------------------------------------
        if False: # 渲染文本
            font = pygame.font.Font(None, 10)
            if True:
                note = note.replace('♯','#') # 无法显示原来的字符'♯'
            text = font.render(note, True, 'white', color)
            # 显示文本
            textRect = text.get_rect()
            textRect.center = (x+rct_note_clr_w/2,rct_note_clr_y+rct_note_clr_h/2)
            screen.blit(text, textRect)
        # ---------------------------------------------------
    # pygame.display.flip()

def show_note_rgbs_dynamic(notes_rgbs_dict,rct,screen,cur_note):
    rct_note_clr_x,rct_note_clr_y,rct_note_clr_w,rct_note_clr_h=rct
    N = len(notes_rgbs_dict)
    rct_note_clr_w=rct_note_clr_w/N
    b_get = False
    for idx,note in enumerate(notes_rgbs_dict.keys()):
        color=notes_rgbs_dict[note]
        x =rct_note_clr_x+rct_note_clr_w*idx
        rect = pygame.Rect(x,rct_note_clr_y, rct_note_clr_w, rct_note_clr_h)
        if note == cur_note:
            b_get=True
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, 'grey', rect,1)
        else:
            pygame.draw.rect(screen, 'black', rect)
    if b_get==False:
        print('cur note note found ={}'.format(cur_note))

def get_all_params_frm_features(hdl_fts,fnwav,fnfeature,mood):
    fts_dict = hdl_fts.extract_all_features(fnwav,fnfeature)
    if fts_dict==None:
        print("current file is not exists")
        os.exit()
    xt,fs = fts_dict["wav"]
    # xt = np.abs(xt)
    xt = xt/max(xt)
    xt_smooth = xt
    # xt_expand = np.pad(xt,(0,44100),'constant',constant_values=(0,0))
    # xt_smooth = savgol_filter(xt_expand, window_length=15, polyorder=2)
    t_pwrs,power = fts_dict["power"]
    t_f0s,f0s= fts_dict["f0s"]
    f0s_notes,f0s_rgbs,notes_rgbs_dict=f0_to_note_and_rgb(f0s,mood)
    t_beats,idx_beats= fts_dict["beats"]
    # beats_rgbs,beats_uniq_colors=beats_to_rgb(idx_beats,mood='excited')
    return [xt_smooth,fs,t_pwrs,power,t_f0s,f0s,f0s_notes,f0s_rgbs,notes_rgbs_dict, t_beats,idx_beats]


running=False
b_exit =False

def show_wav(fnwav,fnfeature,mood):
    global running
    running = True
    # ------------------------------------------------------------
    hdl_fts = song_features()
    res =get_all_params_frm_features(hdl_fts,fnwav,fnfeature,mood)
    xt_smooth,fs,t_pwrs,power,t_f0s,f0s,f0s_notes,f0s_rgbs,notes_rgbs_dict, t_beats,idx_beats=res
    lx =len(xt_smooth)
    nbins,nfrms=power.shape
    # ------------------------------------------------------------
    print('start playing and show music')
    fig_start_x =0
    fig_start_y= 0 # 50
    fig_sub_w=500
    fig_sub_h  = 300
    # ------------------------------------------------------------
    rct_wav_x=fig_start_x
    rct_wav_y=fig_start_y
    rct_wav_w = fig_sub_w
    rct_wav_h = fig_sub_h

    rct_note_x = fig_start_x+fig_sub_w
    rct_note_y=  fig_start_y
    rct_note_w = fig_sub_w
    rct_note_h = fig_sub_h
    h_for_note_clr=20

    rct_spec_x=fig_start_x
    rct_spec_y= fig_start_y+fig_sub_h
    rct_spec_w = fig_sub_w
    rct_spec_h = fig_sub_h

    rct_beats_x=fig_start_x+fig_sub_w
    rct_beats_y=fig_start_y+fig_sub_h
    rct_beats_w = fig_sub_w
    rct_beats_h = fig_sub_h

    win_width= fig_start_x +rct_spec_w+rct_wav_w
    win_height =fig_start_y+ rct_spec_h + rct_wav_h

    # target_area=pygame.Rect(fig_start_x,fig_start_y+150,win_width,100)

    bin_width=rct_spec_w/nbins


    screen = pygame.display.set_mode((win_width, win_height))
    fpath,fname=os.path.split(fnwav)
    ftitle,fext=os.path.splitext(fname)
    pygame.display.set_caption(ftitle)

    pygame.init()
    if True:
        rct_note_clr=[rct_note_x,rct_note_y+rct_note_h-h_for_note_clr,rct_note_w,h_for_note_clr]
        # rct_note_clr=[0,0,win_width,30]
        show_note_rgbs(notes_rgbs_dict,rct_note_clr,screen)
        pygame.display.flip()

    # 加载音乐
    pygame.mixer.init(frequency=fs, size=-16, channels=2, buffer=2048)
    pygame.mixer.music.load(fnwav)
    pygame.mixer.music.play()
    # clock = pygame.time.Clock()
    # cnt_arr=[]
    # font=pygame.font.SysFont(None,36)
    # font = pygame.font.Font('songti.ttc',20) # /System/Library/Fonts/Supplemental
    font = pygame.font.Font('/System/Library/Fonts/Supplemental/Songti.ttc',20) #
    line_wav_color = (0, 255, 0)
    line_spec_color=(255,255,0)
    base_bkg_clr = 'black'
    font_clr = 'white'

    rect_beats = pygame.Rect(rct_beats_x,rct_beats_y, rct_beats_w, rct_beats_h)
    rect_wav=pygame.Rect(rct_wav_x,rct_wav_y, rct_wav_w, rct_wav_h)
    rect_note = pygame.Rect(rct_note_x,rct_note_y, rct_note_w, rct_note_h-h_for_note_clr)
    rect_spec=pygame.Rect(rct_spec_x,rct_spec_y, rct_spec_w, rct_spec_h)
    tms_pre_wav=0
    tms_pre_note=0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.mixer.music.stop()
                pygame.quit()
                b_exit = True
                return
        # if is_playing:
        if pygame.mixer.music.get_busy():

            # 获取音乐样点
            tcur_ms = pygame.mixer.music.get_pos()
            # ---------------------------------------------------
            if True:   # 显示 节拍
                i_beat = np.argmin(np.abs(t_beats-tcur_ms))
                trg_beat= np.min(np.abs(t_beats-tcur_ms))
                if False:
                    beats_bkg_color=beats_rgbs[i_beat]
                    pygame.draw.rect(screen, beats_bkg_color, rect_beats)
                else:
                    beats_bkg_color=base_bkg_clr
                    pygame.draw.rect(screen, beats_bkg_color, rect_beats)
                    # nbu=len(beats_uniq_colors)
                    nbu=np.max(idx_beats)-np.min(idx_beats)+1
                    vv = 10
                    s = rct_beats_x+vv
                    w = (rct_beats_w-vv*2)/(nbu-1)
                    r = min(8,w * 0.1)
                    ball_clr='white'
                    if True:  # 重音没变
                        if trg_beat<150: # 持续时间不到 单位ms，绘图内容不变
                            for k in range(nbu):
                                idb =  k+1 #节拍号标注是1，2，3，...
                                x = s + k*w
                                if idb== idx_beats[i_beat]:
                                    if idb==1: #第一个视为重音，其它视为轻拍
                                        y = rct_beats_y+r+rct_beats_h*0.1
                                    else: #轻拍
                                        y = rct_beats_y+r+rct_beats_h*0.5
                                    pygame.draw.circle(screen,ball_clr, (x,y),r)
                                else:
                                    y = rct_beats_y+rct_beats_h-r
                                    pygame.draw.circle(screen, ball_clr, (x,y),r)
                                if k>0:
                                    pygame.draw.line(screen, ball_clr,(xpre,ypre),(x,y),2)
                                xpre  =x
                                ypre=y
                                # print('x,y={},{}',x,y)

                        else: #同一个beat，持续时间超过限定时间，置零
                            for k in range(nbu):
                                idb = k+1  #节拍号标注是1，2，3，...
                                x = s + k*w
                                y = rct_beats_y+rct_beats_h-r
                                pygame.draw.circle(screen, ball_clr, (x,y),r)
                                if k>0:
                                    pygame.draw.line(screen, ball_clr,(xpre,ypre),(x,y),2)
                                xpre  =x
                                ypre=y

                str_beats_title='显示节拍' #'show beats'

                text = font.render(str_beats_title, True, font_clr, beats_bkg_color)            #
                textRect = text.get_rect()
                textRect.center = (rct_beats_x+textRect.width/2,rct_beats_y+20)
                screen.blit(text, textRect)

            # ---------------------------------------------------
            if tcur_ms-tms_pre_wav>100: #True: # 显示波形
                tms_pre_wav=tcur_ms
                bkg_wav_color=base_bkg_clr
                i_smp_start=int(tcur_ms/1000*fs) -  rct_wav_w//2 #样点
                pygame.draw.rect(screen, bkg_wav_color, rect_wav)
                for k in range(rct_wav_w):
                    idx = k*2+i_smp_start #k+i_smp_start
                    if idx>=lx or idx <0 :
                        y=0
                    else:
                        y=xt_smooth[idx]*rct_wav_h//2
                    x=rct_wav_x+k
                     # 全波整流
                    ypos = rct_wav_y+rct_wav_h//2 - y

                    y_width=abs(y)

                    rect = pygame.Rect(x,ypos, 1,y_width)
                    pygame.draw.rect(screen, line_wav_color, rect)


                str_wav_title='显示波形' #'show wave'
                text = font.render(str_wav_title, True, font_clr, bkg_wav_color)            #
                textRect = text.get_rect()
                textRect.center = (rct_wav_x+textRect.width/2,rct_wav_y+20)
                screen.blit(text, textRect)
            # ---------------------------------------------------

            if tcur_ms-tms_pre_note>100: #True: # 背景颜色反应note
                tms_pre_note=tcur_ms
                f0_pos=np.argmin(np.abs(t_f0s-tcur_ms))
                bkg_note_color = f0s_rgbs[f0_pos]
                cur_note = f0s_notes[f0_pos]
                # str_note='show note: %s'% cur_note.replace('♯','#') # 无法显示原来的字符'♯'
                str_note='显示音符: %s'% cur_note.replace('♯','#')
                pygame.draw.rect(screen, base_bkg_clr, rect_note)

                rg_note_h=40
                rct_notes_clr=[rct_note_x,rct_note_y+rg_note_h,rct_note_w,rct_note_h-h_for_note_clr-rg_note_h]
                show_note_rgbs_dynamic(notes_rgbs_dict,rct_notes_clr,screen,cur_note)


                text = font.render(str_note, True, font_clr, base_bkg_clr)            #
                textRect = text.get_rect()
                textRect.center = (rct_note_x+textRect.width/2,rct_note_y+20)
                screen.blit(text, textRect)
            # ---------------------------------------------------
            # 显示频谱
            if True:
                bkg_spec_color=base_bkg_clr
                pygame.draw.rect(screen,bkg_spec_color, rect_spec)
                pwr_pos=np.argmin(np.abs(t_pwrs-tcur_ms))
                pwr_spec=power[:,pwr_pos]
                if False:  # 柱状图
                    for k in range(nbins):
                        x=rct_spec_x+ k*bin_width
                        y=pwr_spec[k]*rct_spec_h # +5 #坐标方向向下
                        ypos = rct_spec_y+  rct_spec_h -y
                        rect = pygame.Rect(x,ypos, bin_width, y)
                        pygame.draw.rect(screen, tuple(line_spec_color), rect)
                else:
                    xpre=rct_spec_x
                    ypre=rct_spec_y+  rct_spec_h - pwr_spec[0]*rct_spec_h #
                    for k in range(1,nbins):
                        x=rct_spec_x+ k*bin_width
                        y=pwr_spec[k]*rct_spec_h # +5 #坐标方向向下
                        y = rct_spec_y+  rct_spec_h -y
                        pygame.draw.line(screen, tuple(line_spec_color),(xpre,ypre),(x,y),2)
                        xpre = x
                        ypre = y

                str_spec_title='显示频谱' #'show spectrum'

                text = font.render(str_spec_title, True, font_clr, bkg_spec_color)            #
                textRect = text.get_rect()
                textRect.center = (rct_spec_x+textRect.width/2,rct_spec_y+20)
                screen.blit(text, textRect)

            pygame.display.flip()

            # print("time={}, bkg={}, note={}".format(tcur_ms,bkg_spec_color,str_note))
        else:
            running = False
            pygame.mixer.music.stop()
            screen.fill(base_bkg_clr)
            # str_message='Please refer to commond line ,to choose your next operation'

            str_message=['请从命令行输入下一个需要显示的歌曲以及情感，中间用空格隔开'] #,'如需要展示的歌曲为"白龙马"，情感为"激昂"可输入：','白龙马 激昂']
            nmsg = len(str_message)
            for k in range(nmsg):
                text = font.render(str_message[k], True, font_clr, base_bkg_clr)            #
                textRect = text.get_rect()
                textRect.center =(win_width//2,win_height//2+k*50)
                screen.blit(text, textRect)

            # pygame.draw.rect(screen,'white', target_area)
            pygame.display.flip()
            # key_check()
            break
    # 停止音乐
    # del screen
    pygame.quit()


import threading

def key_check():
    global running
    # global prev_fnwav
    # global prev_fnfeature
    global prev_ftitle
    global prev_mood
    global datadir
    while True:
        if running==False:
            strinfo2="请输入需要展示的音频文件以及对应的情感（空格隔开）\n"
            strinfo2+="\t情感与颜色的对应关系如下:\n"
            strinfo2=strinfo2+"\thappy or 欢快- 橙色到绿色\n"
            strinfo2=strinfo2+"\tcalm or 冷静- 蓝绿色到紫色\n"
            strinfo2=strinfo2+"\texcited or 激昂- 红色到黄色\n"
            strinfo2=strinfo2+"\tsad or 伤感- 蓝色到蓝灰色\n"
            strinfo2=strinfo2+"\tsweet or 甜美- 粉色到浅粉色\n"
            strinfo2=strinfo2+"\t默认的歌曲为:{}, 情感为:{}，若使用默认情感，可以直接输入'回撤'按键\n".format(prev_ftitle,prev_mood)
            res = input(strinfo2)
            # print('res=%s'%res)
            if res=='q':
                print('退出当前任务')
                break
            if res=='': #直接回车
                ftitle = prev_ftitle
                mood=prev_mood
            else:
                arr=res.split()
                la = len(arr)
                if la>=1:
                    ftitle=arr[0]

                if la>=2:
                    mood = arr[1]
                else:
                    mood = prev_mood
            fnwav ='%s/%s.mp3'%(datadir,ftitle)
            fnfeature = '%s/%s.npy'%(datadir,ftitle)
            if os.path.exists(fnwav)==False:
                print("文件:%s 不存在"%fnwav)

            print('show wav')
            show_wav(fnwav,fnfeature,mood)
            prev_ftitle = ftitle
            prev_mood = mood


datadir='./data'
prev_ftitle = 'libai'
prev_ftitle = '白龙马'
prev_mood='happy'
ftitles=['jingle_bells','little_star','symphony_5']
prev_ftitle=ftitles[0]

if __name__=='__main__':
    ftitle= prev_ftitle
    mood= prev_mood
    fnwav ='%s/%s.mp3'%(datadir,ftitle)
    fnfeature = '%s/%s.npy'%(datadir,ftitle)
    show_wav(fnwav,fnfeature,mood)
    # print('runing state: {}'.format(running))
    # thr = threading.Thread(target=key_check)
    # thr.start()
    # thr.join()
    key_check()



