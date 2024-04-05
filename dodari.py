import os, time, datetime, platform
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nltk
nltk.download('punkt', quiet=True)
import gradio as gr
from langdetect import detect

import logging
logging.getLogger().disabled = True 
logging.raiseExceptions = False
import warnings
warnings.filterwarnings('ignore')

class Dodari:
    def __init__(self):
        self.max_len = 512
        self.selected_files = []
        self.lang_opt = ["한국어", "영어", "일본어"]
        self.lang_list = ['kor_Hang', 'eng_Latn', 'jpn_Jpan' ]
        self.origin_lang = None
        self.target_lang = "kor_Hang"
        self.origin_lang_str = None
        self.target_lang_str = self.lang_opt[self.lang_list.index(self.target_lang)]
        self.model_list = ['facebook/nllb-200-distilled-600M', 'facebook/nllb-200-distilled-1.3B', 'facebook/nllb-200-3.3B' ]
        self.model_opt = ["4GB 이하 - 이해할만한 퀄임. 가끔 개소리가 섞임", "4GB ~ 8GB - 가성비 좋은 퀄. 어쩔땐 꽤 좋음", "8GB이상 - 퀄이 꽤 좋아짐. 속도가 더 빨라지는건 아님"]
        self.selected_model = 'facebook/nllb-200-distilled-1.3B'
        self.model = None
        self.tokenizer = None
        self.global_trans_script = None
        self.css = """
            .radio-group .wrap {
                display: float !important;
                grid-template-columns: 1fr 1fr;
            }
            """
        self.start = '' 
        self.platform = platform.system()

    def main(self):
        
        with gr.Blocks(css=self.css, theme=gr.themes.Default(primary_hue="red", secondary_hue="pink")) as app:
            gr.HTML("<h1>다국어 AI번역기 '<span style='color:red'>도다리</span>' ver. NLLB 입니다 </h1>")
            gr.HTML("<p>영한/한영 번역에 특화된 <a target='_blank' href='https://github.com/vEduardovich/dodari'>새로운 도다리</a>를 사용해보세요. *.txt뿐만 아니라 *.epup 전자책 번역도 가능합니다.</p>")
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    with gr.Tab('순서 1'):
                        gr.Markdown("<h3>1. 번역이 필요한 파일들 선택</h3>")
                        input_window = gr.File(file_count="files", label='파일들')

                with gr.Column():
                    with gr.Tab('순서 2'):
                        gr.HTML("<h3>2. 번역 언어 선택</h3>")
                        
                        origin = gr.Textbox(label="원본", value=None)
                        gr.HTML("<span style='display:block;font-size:2em;text-align:center;'>⬇</span> ")
                        target = gr.Radio( choices=self.lang_opt, label='타겟', value=self.target_lang_str)
                        
                        
                        input_window.change(fn=self.change_upload, inputs=input_window, outputs=origin, preprocess=False)
                        
                        target.change(fn=self.change_target_lang, inputs=target, outputs=None)
                with gr.Column():
                    with gr.Row():
                        with gr.Tab('순서 3'):
                            gr.HTML("<span style='display:flex;'><h3>3. 번역모델 선택</h3> <span style='margin-top:15px;margin-left:10px;color:grey;'>- 번역의 품질을 결정합니다</span></span>")
                            
                            selected_model = gr.Radio(elem_classes="radio-group", choices=self.model_opt, value="4GB ~ 8GB - 가성비 좋은 퀄. 어쩔땐 꽤 좋음", label='자신이 가진 그래픽 카드 메모리(VRAM)에 맞춰 선택합니다')
                            selected_model.change(fn=self.change_model, inputs=selected_model)
                            gr.HTML("<div style='text-align:right'><p style = 'color:grey;'>처음 실행시 모델을 다운받는데 아주 오랜 시간이 걸립니다.</p><p style='color:grey;'>컴퓨터 사양이 좋다면 번역 속도가 빨라집니다.</p><p style='color:grey;'>맥m1이상에서는 mps를 이용하여 가속합니다</p></div>")
                            graphic_script = "<a href='https://www.google.com/search?q=%EC%9C%88%EB%8F%84%EC%9A%B0+%EA%B7%B8%EB%9E%98%ED%94%BD++%EB%A9%94%EB%AA%A8%EB%A6%AC+%ED%99%95%EC%9D%B8%ED%95%98%EB%8A%94+%EB%B2%95' target='_blank' style='display:block;text-align:right;'>내 그래픽 카드 메모리 확인하는 법</a>"
                            gr.HTML(graphic_script)
            with gr.Tab('순서 4'):
                translate_btn = gr.Button(value="번역 실행하기", size='lg', variant="primary")
                with gr.Row():
                    
                    msg = gr.Textbox(label="상태 정보", scale=4, value='번역 대기중..')
                    translate_btn.click(fn=self.translateFn, outputs=msg)
                    btn_openfolder = gr.Button(value='📂 번역 완료한 파일들 보기', scale=1, variant="secondary")
                    btn_openfolder.click(fn=lambda: self.open_folder(), inputs=None, outputs=None)

        app.launch(inbrowser=True)
    def translateFn(self, progress=gr.Progress()):
        if not self.selected_files : return "번역할 파일을 추가하세요."
        elif not self.origin_lang : return "원본 언어를 선택하세요."
        elif not self.target_lang : return "타겟 언어를 선택하세요."
        elif not self.selected_model : return "번역 모델을 선택해주세요."
        
        self.start = time.time()
        progress(0, desc="번역 모델을 준비중입니다...")

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" )

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.selected_model, cache_dir=os.path.join("models", "tokenizers"))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=self.selected_model, cache_dir=os.path.join("models"))

        translator = pipeline('translation', model=self.model, tokenizer=self.tokenizer, device= device, src_lang=self.origin_lang, tgt_lang=self.target_lang, max_length=self.max_len)

        for file in progress.tqdm(self.selected_files, desc='파일전체'):
            
            origin_abb = self.origin_lang.split(sep='_')[0]
            target_abb = self.target_lang.split(sep='_')[0]
            
            name, ext = os.path.splitext(file['orig_name'])
            output_file_bi = self.write_filename( "{name}_{t2}({t3}).{ext}".format(name=name, t2=target_abb, t3=origin_abb, ext = ext) )
            output_file = self.write_filename( "{name}_{t2}.{ext}".format(name=name, t2=target_abb, ext = ext) )

            book = self.get_filename(file['path']);
            book_list = book.split(sep='\n')

            book_list = book.split(sep='\n')
            for book in progress.tqdm(book_list, desc='단락'):
                particle = nltk.sent_tokenize(book)
                
                for text in progress.tqdm( particle, desc='문장' ):
                    output = translator(text, max_length=self.max_len)
                    output_file_bi.write("{t1} ({t2})".format(t1=output[0]['translation_text'], t2=text) )
                    output_file.write(output[0]['translation_text'])
                output_file_bi.write('\n')
                output_file.write('\n')

        
        output_file_bi.close()
        output_file.close()
        sec = self.check_time()
        self.start = None
        
        return "번역완료! 걸린시간 : {t1}".format(t1=sec)

    def change_upload(self, files):
        try:
            self.selected_files = files
            if not files : return gr.Textbox(label="원본", value="번역할 파일을 먼저 추가해주세요")
            aBook = files[0]

            book = self.get_filename(aBook['path']);
            check_lang = detect(book[0:200])
            self.origin_lang_str = self.lang_opt[1] if 'en' in check_lang else self.lang_opt[0] if 'ko' in check_lang else self.lang_opt[2]
            self.origin_lang = self.lang_list[1] if 'en' in check_lang else self.lang_list[0] if 'ko' in check_lang else self.lang_list[2]
            
            return gr.Textbox(label="원본", value=self.origin_lang_str)
            
        except Exception as err:
            return gr.Textbox(label="원본", value=None)

    def change_origin_lang(self, lang):
        self.origin_lang = self.lang_list[0] if lang == '한국어' else self.lang_list[1] if lang=='영어' else self.lang_list[2]
        self.global_trans_script = "<span style='color:skyblue;font-size:1.5em;'>{t1}</span><span>를 </span>".format(t1=lang)

        if self.target_lang: 
            t2 = self.lang_opt[0] if self.target_lang == self.lang_list[0] else self.lang_opt[1] if self.target_lang == self.lang_list[1] else self.lang_opt[2]
            return "{t1} <span style='color:red;font-size:1.5em;'> {t2}</span><span>로 번역합니다.</span>".format(t1=self.global_trans_script, t2=t2)
        else:
            return self.global_trans_script

    def change_target_lang(self, lang):
        self.target_lang = self.lang_list[0] if lang == '한국어' else self.lang_list[1] if lang=='영어' else self.lang_list[2]
        same_lang = ''
        if self.origin_lang == self.target_lang:
            same_lang = "<span style='font-size:0.9em;color:grey'> (이게 무슨 의미가 있나요)</span>"
        return "{t1} <span style='color:red;font-size:1.5em;'> {t2}</span><span>로 번역합니다.</span> {t3}".format(t1=self.global_trans_script, t2=lang, t3=same_lang)

    def change_model(self, model):
        self.selected_model = self.model_list[0] if model == self.model_opt[0] else self.model_list[1] if model == self.model_opt[1] else self.model_list[2]

    def get_filename(self, fileName):
        try:
            input_file = open(fileName, 'r', encoding='utf-8')
            return input_file.read()
        except:
            try :
                input_file = open(fileName, 'r', encoding='euc-kr')
                return input_file.read()
            except :
                input_file = open(fileName, 'r', encoding='cp949', errors='ignore')
                return input_file.read()
            
    def write_filename(self, file):
        saveDir = os.path.join(os.getcwd(), 'outputs')
        if not(os.path.isdir(saveDir)): 
            os.makedirs(os.path.join(saveDir)) 

        file_name = saveDir + '/' + file
        output_file = open(file_name, 'w', encoding='utf-8')
        return output_file

    def open_folder(self):
        saveDir = os.path.join(os.getcwd(), 'outputs')
        if not(os.path.isdir(saveDir)): 
            os.makedirs(os.path.join(saveDir)) 
        if  self.platform == 'Windows': os.system(f"start {saveDir}")
        elif self.platform == 'Darwin': os.system(f"open {saveDir}")
        elif self.platform == 'Linux': os.system(f"nautilus {saveDir}")
        
    def check_time(self):
        end = time.time()
        during = end - self.start
        sec = str(datetime.timedelta(seconds=during)).split('.')[0]
        return sec
if __name__ == "__main__":
    dodari = Dodari()
    dodari.main()
