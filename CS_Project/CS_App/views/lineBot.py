# lineBot.py
from django.http import HttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage,TextSendMessage,QuickReply,QuickReplyButton, MessageAction ,TemplateSendMessage,ButtonsTemplate,URIAction,PostbackAction,CarouselTemplate, CarouselColumn,FlexSendMessage,Postback,BoxComponent,BubbleContainer,AudioMessage,AudioSendMessage
from linebot.models import BubbleStyle,ButtonComponent,TextComponent,BlockStyle # LINE 訊息事件與訊息類型
from ..models import Line_Feedback,LineUser
from module_slef.module_output import ask_rag_key,ask_cag
#from ..module_slef.module_keyRAG_v7 import ask_rag as ask_rag_key
#from ..module_slef import ask_rag as ask_rag_main
from module_slef.module_RAG import ask_rag
from module_slef.module_unRAG import ask_direct
from module_slef.J1_module_KGRAG import ask_enhanced_hybrid_rag as ask_kgrag
from module_slef.J2_module_GraphRAG_v1 import ask_question as ask_graphrag
from module_slef.J3_module_StuRAG import ask_question as ask_sturag
from module_slef.module_CAG import init_cag_module
init_cag_module()
import re
import speech_recognition as sr
from pydub import AudioSegment
import os
from gtts import gTTS
import io
import logging
import cloudinary
import cloudinary.uploader



# 假設這裡的 chain 來自於某個 AI 模組，但原始碼中沒有提供
#from .module_slef.some_module import chain




line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(settings.LINE_CHANNEL_SECRET)
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "auto").lower() # auto / openai / gtts

cloudinary.config(
    cloud_name="dybhscnbv",
    api_key="247224169944335",
    api_secret="Q67ryhLgtMKDJJ24xLFbQCoFv4o"
)

# 創建按鈕 Action
link_action = PostbackAction(
    label='綁定帳號',  # 按鈕上顯示的文字 (對應 C# 的 label)
    data='Link'      # 點擊後傳回伺服器的資料 (對應 C# 的 data)
    # 這裡可以選擇不加 display_text，LINE 預設會顯示 label
)


# 創建 Flex Message
flex_message_content = BubbleContainer(
    # 訊息主體 Body
    body=BoxComponent(
        layout='vertical',
        contents=[
            TextComponent(
                text='尚未綁定帳號',
                weight='bold',
                size='md',
                align='center'
            )
        ]
    ),
    # 訊息底部 Footer
    footer=BoxComponent(
        layout='vertical',
        contents=[
            ButtonComponent(
                style='primary',
                action=link_action
            )
        ]
    ),
    # 設定樣式 (Styles)
    styles=BubbleStyle(
        footer=BlockStyle(
            separator=True  # 底部加上分隔線
        )
    )
)


    # 最終傳送的 Flex 訊息物件
    #flex_message = FlexSendMessage(
     #   alt_text='尚未綁定帳號',  # 替代文字
      #  contents=flex_message_content
    #)
def convert_to_text(messages):
    # 單一 TextSendMessage 物件
    if isinstance(messages, TextSendMessage):
        return messages.text

    # 如果是 list
    if isinstance(messages, list):
        collected = []
        for m in messages:
            if isinstance(m, TextSendMessage):
                collected.append(m.text)
            else:
                collected.append(str(m))  # 防止其他型別造成錯誤
        return " ".join(collected)

    # 若完全不是文字訊息（防呆）
    return str(messages)


def clean_markdown_format(text: str) -> str:
    """
    清除文字中的 Markdown 格式標記，包括 *, **, #, 並處理不必要的空白和換行。

    Args:
        text (str): 含有 Markdown 格式的輸入字串。

    Returns:
        str: 清除格式後的字串。
    """
    # 1. 移除 Markdown 列表符號 (*, -, +) 和標題符號 (#)
    #    - 注意：這裡我們只針對常見的 *, ** 和 #
    #    - 我們需要處理列表符號後可能跟隨的空白
    text = re.sub(r'^\s*[\*\-+#]+\s*', '', text, flags=re.MULTILINE)

    # 2. 移除粗體標記 (**, *)
    text = re.sub(r'[\*\*]', '', text)
    text = re.sub(r'[\*]', '', text)

    # 3. 移除 Markdown 區塊引號 (>) - 如果有
    text = re.sub(r'^\s*>\s*', '', text, flags=re.MULTILINE)

    # 4. 移除多餘的空白行和行首行尾的空白
    #    - 將多個連續換行替換為單個換行（或空格，視需求而定）
    text = re.sub(r'(\n\s*)+\n', '\n', text)
    
    # 5. 移除連續的空格，並將換行符替換為單一空格（保持可讀性）
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. 將列表/項目之間的換行再替換回來，以保持段落結構
    #    (因為您的原始輸入中列表是分行的，我們嘗試保留這個結構)
    #    這裡的處理策略是：先全部變成單行，然後再用一個換行符分隔內容。
    #    如果希望每個項目獨立一行，可以對步驟5的結果做進一步處理。

    # 基於您的輸入，我們將主要內容重新以換行分隔
    # 輸入: 根據提供的資訊，國立聯合大學目前有以下校區： * 第一（二坪山）校區 * 第二（八甲）校區 此外，資料中也提到了「國立聯合大學職業安全衛生教育訓練管理要點」以及「校務基金工作人員契約書」，但這些並非學院名稱。
    # 輸出: 根據提供的資訊，國立聯合大學目前有以下校區：
    #       第一（二坪山）校區
    #       第二（八甲）校區
    #       此外，資料中也提到了「國立聯合大學職業安全衛生教育訓練管理要點」以及「校務基金工作人員契約書」，但這些並非學院名稱。
    
    # 由於您的輸入是以一個大段落形式呈現的，我會盡量讓它保持單行，但去除標記。
    # 如果您希望每個項目單獨一行，請看下面的示例執行結果。
    
    return text

@csrf_exempt
def line_webhook(request):
   if request.method == "POST":
       signature = request.headers.get("X-Line-Signature")
       body = request.body.decode("utf-8")
       try:
           handler.handle(body, signature)
       except InvalidSignatureError:
           return HttpResponse(status=400)
       return HttpResponse("OK")
   return HttpResponse("This endpoint is for LINE Webhook only.")

last_reply = {}

@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    # 1. 建立一個列表來儲存所有要回覆的訊息
    reply_messages = []
    # 2. 加入收到訊息的確認（取代了錯誤的 event.message.append）
    reply_messages.append(TextSendMessage(text='已收到您的聲音訊息，正在進行語音轉文字處理...'))

    # 準備檔案路徑
    download_path = './static/temp_audio'
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Line 傳送的檔案類型不固定，使用原始 ID 作為檔名的一部分
    original_path = os.path.join(download_path, f'{event.message.id}.m4a')
    wav_path = os.path.join(download_path, f'{event.message.id}.wav')

    try:
        # 3. 下載音檔
        audio_content = line_bot_api.get_message_content(event.message.id)
        with open(original_path, 'wb') as fd:
            for chunk in audio_content.iter_content():
                fd.write(chunk)

        # 4. 語音轉文字處理
        r = sr.Recognizer()
        
        # 確保 ffmpeg 路徑設定正確，如果您的 ffmpeg 在環境變數中，這行可以省略
        AudioSegment.converter = 'C:\\ffmpeg\\bin\\ffmpeg.exe' 
        
        # 將原始音檔轉換為 WAV 格式
        sound = AudioSegment.from_file(original_path)
        sound.export(wav_path, format="wav")
        
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
            
        text = r.recognize_google(audio, language='zh-Hant') # 進行語音辨識
        
        # 5. 將轉換的文字加入回覆列表
        reply_messages.append(TextSendMessage(text=f"🗣️ 語音辨識結果：\n{text}"))

    except sr.UnknownValueError:
        # 語音無法辨識的錯誤
        reply_messages.append(TextSendMessage(text='很抱歉，無法辨識您說的內容。'))
    except Exception as e:
        # 其他處理錯誤 (例如 ffmpeg 轉換失敗, 檔案操作錯誤等)
        error_msg = f"語音處理失敗，請檢查設定或檔案：{e}"
        reply_messages.append(TextSendMessage(text=error_msg))
        print(error_msg)
    finally:
        # 6. 清理暫存檔案，不論成功或失敗都嘗試刪除
        if os.path.exists(original_path):
            os.remove(original_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

    # 7. 使用 line_bot_api.reply_message 回覆整個訊息列表
    # 這取代了您舊的 line_bot_api.reply_message(event.reply_token, event.message)
    line_bot_api.reply_message(event.reply_token, reply_messages)
    return

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
   user_id = event.source.user_id
   text = event.message.text.strip()
   user, _ = LineUser.objects.get_or_create(line_user_id=user_id)
   model = user.preferred_model
   response=None
   current_reply=[]
   related_links = []
   if text.startswith("使用方法："):
       selected_model = text.replace("使用方法：", "").strip()
       user.preferred_model = selected_model
       user.save()
       line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"方法已切換為：{selected_model}"))
       return
   
   if text=="語音回覆":
        if user_id not in last_reply:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="沒有可轉成語音的內容喔！")
            )
            return

        # 取得上次的文字
        text_to_tts = last_reply[user_id]
        #print(type(text_to_tts), text_to_tts)
        if isinstance(text_to_tts, list):
            text_to_tts = " ".join(text_to_tts)
        elif not isinstance(text_to_tts, str):
            text_to_tts = str(text_to_tts)
        # 生成語音檔
        tts = gTTS(text=text_to_tts, lang='zh-TW')
        file_path = f"./static/temp_audio/{user_id}_reply.mp3"
        tts.save(file_path)
        
        upload_result = cloudinary.uploader.upload(
            file_path,
            resource_type="video"  # mp3 要用 "video"
        )
        public_url = upload_result["secure_url"]
        print(public_url)
        # 上傳語音檔給 LINE
        message = AudioSendMessage(
            original_content_url=public_url,
            duration=5000  # 毫秒
        )
        line_bot_api.reply_message(event.reply_token, message)
        return
       


   # 若使用者在回饋模式中
   if user.in_feedback_mode:
       # 儲存回饋
       Line_Feedback.objects.create(line_user=user, content=text)
       user.in_feedback_mode = False
       user.save()




       reply = TextSendMessage(text="感謝您的寶貴意見，我們已收到您的回饋！")
       line_bot_api.reply_message(event.reply_token, reply)
       return




   # 若使用者說「我要回饋」，啟用回饋模式
   if text =="意見信箱":
       user.in_feedback_mode = True
       user.save()
       reply = TextSendMessage(text="請輸入您想回饋的內容，我們非常樂意聽取您的建議。")
       line_bot_api.reply_message(event.reply_token, reply)
       return
 
   #指令：顯示模型選擇
   if text.lower() == "方法更改":
       """reply = TemplateSendMessage(
           alt_text='ButtonsTemplate',
           template=ButtonsTemplate(
           thumbnail_image_url='https://media.discordapp.net/attachments/1423258585621856372/1440930971829403682/Gemini_Generated_Image_jpcwycjpcwycjpcw.jpg?ex=691ff30a&is=691ea18a&hm=28db3616574392f66f0bbbf8f9bc8b4d40b176628b11e76a19083b1f05754214&=&format=webp&width=649&height=649',
           title='方法更改',
           text='請選擇檢索方法',
           actions=[
               MessageAction(
                   label='資料結構化檢索',
                   text='使用方法：資料結構化檢索'
               ),
               MessageAction(
                   label='關鍵詞檢索',
                   text='使用方法：關鍵詞檢索'
               ),
               MessageAction(
                   label='關鍵詞擴充檢索',
                   text='使用方法：關鍵詞擴充檢索'
               ),
               MessageAction(
                   label='廣度檢索',
                   text='使用方法：廣度檢索'
               )
           ]
       )
   )
       line_bot_api.reply_message(event.reply_token, reply)
       return"""
       carousel_template_message = TemplateSendMessage(
            alt_text='Carousel template',
            template=CarouselTemplate(
                    columns=[
                        CarouselColumn(
                            thumbnail_image_url='https://cdn.discordapp.com/attachments/1423258585621856372/1441350267571273738/Gemini_Generated_Image_jpcwycjpcwycjpcw.jpg?ex=6921798a&is=6920280a&hm=9e1bbc3dcd0cfb8e63e35dff14397de589804d9d6df37e87dfacaa7394e53ab8&',
                            title='方法更改',
                            text='請選擇檢索方法',
                            actions=[
                                MessageAction(
                                    label='一般檢索',
                                    text='使用方法：一般檢索'
                                ),
                                MessageAction(
                                    label='關鍵詞檢索',
                                    text='使用方法：關鍵詞檢索'
                                ),
                                MessageAction(
                                    label='關鍵詞擴充檢索',
                                    text='使用方法：關鍵詞擴充檢索'
                                )
                            ]
                        ),
                        CarouselColumn(
                            thumbnail_image_url='https://cdn.discordapp.com/attachments/1423258585621856372/1441350267571273738/Gemini_Generated_Image_jpcwycjpcwycjpcw.jpg?ex=6921798a&is=6920280a&hm=9e1bbc3dcd0cfb8e63e35dff14397de589804d9d6df37e87dfacaa7394e53ab8&',
                            title='方法更改',
                            text='請選擇檢索方法',
                            actions=[
                                MessageAction(
                                    label='廣度檢索',
                                    text='使用方法：廣度檢索'
                                ),
                                MessageAction(
                                    label='快速檢索',
                                    text='使用方法：快速檢索'
                                ),
                                MessageAction(
                                    label='資料結構化檢索',
                                    text='使用方法：資料結構化檢索'
                                )
                            ]
                        )
                    ]
                )
        )
       line_bot_api.reply_message(event.reply_token, carousel_template_message)
       return
   if text.lower() == "功能介紹":
     flex_message = {
        "type": "carousel",
        "contents": [
            {
                "type": "bubble",
                "size": "mega",
                "direction": "ltr",
                "header": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {"type": "text", "text": "1. 介紹", "align": "center"}
                    ]
                },
                "hero": {
                    "type": "image",
                    "url": "https://cdn.discordapp.com/attachments/1423258585621856372/1441321125232644137/Gemini_Generated_Image_u7qxkyu7qxkyu7qx.png?ex=69215e66&is=69200ce6&hm=2db4489d3c609d847627f0cfe7ba8d53f716537e1eb4dd35b875b7677600e78a&",
                    "size": "full",
                    "aspectRatio": "1.51:1",
                    "aspectMode": "fit"
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {"type": "text", "text": "這是一個提供聯大校園資訊的", "align": "start"},
                        {"type": "text", "text": "LINE Bot，裡面搭載AI對話系", "align": "start"},
                        {"type": "text", "text": "統供使用者問答。需要使用其", "align": "start"},
                        {"type": "text", "text": "他功能請使用功能選單。", "align": "start"}
                    ]
                },
                "footer": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {"type": "separator"}
                    ]
                }
            },
            {
                "type": "bubble",
                "size": "mega",
                "direction": "ltr",
                "header": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {"type": "text", "text": "2. 方法更改", "align": "center"}
                    ]
                },
                "hero": {
                    "type": "image",
                    "url": "https://cdn.discordapp.com/attachments/1423258585621856372/1441350267571273738/Gemini_Generated_Image_jpcwycjpcwycjpcw.jpg?ex=6921798a&is=6920280a&hm=9e1bbc3dcd0cfb8e63e35dff14397de589804d9d6df37e87dfacaa7394e53ab8&",
                    "size": "full",
                    "aspectRatio": "1.51:1",
                    "aspectMode": "fit"
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {"type": "text", "text": "如果需要不同風格的回應，可點選", "align": "start"},
                        {"type": "text", "text": "功能選單中的[方法更改]。", "align": "start"}
                    ]
                },
                "footer": {
                    "type": "box",
                    "layout": "horizontal",
                    "contents": [
                    {
                        "type": "button",
                        "action": {
                        "type": "uri",
                        "label": "點我觀看示意圖",
                        "uri": "https://cdn.discordapp.com/attachments/1423258585621856372/1441382717982773328/IMG_5780.png?ex=692197c3&is=69204643&hm=6cd8a74f200dcb7e0fbc59cdd2551b611be50bdc0980493da1768788d1fdebe2&"
                        }
                    }
                    ]
                }
            },
            {
                "type": "bubble",
                "direction": "ltr",
                "header": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {"type": "text", "text": "3. 意見信箱", "align": "center"}
                    ]
                },
                "hero": {
                    "type": "image",
                    "url": "https://cdn.discordapp.com/attachments/1423258585621856372/1441321124813340786/Gemini_Generated_Image_jn480rjn480rjn48.png?ex=69215e66&is=69200ce6&hm=b2384997fd73c088a4a3b808afaa14ec7ed40253367a7d9bc3ce12837a8211a9&",
                    "size": "full",
                    "aspectRatio": "1.51:1",
                    "aspectMode": "fit"
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {"type": "text", "text": "點選[意見信箱]可將回饋回傳到開", "align": "start"},
                        {"type": "text", "text": "發端。", "align": "start"}
                    ]
                },
                "footer": {
                    "type": "box",
                    "layout": "horizontal",
                    "contents": [
                    {
                        "type": "button",
                        "action": {
                        "type": "uri",
                        "label": "點我觀看示意圖",
                        "uri": "https://cdn.discordapp.com/attachments/1423258585621856372/1441382717542629438/IMG_5782.png?ex=692197c3&is=69204643&hm=6760dde374c43c2bf1029fa865d262aeebf4907efce4639c7fedfddd8452a8ca&"
                        }
                    }
                    ]
                }
            },
            {
              "type": "bubble",
                "direction": "ltr",
                "header": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                    {
                        "type": "text",
                        "text": "4. 進入平台",
                        "align": "center",
                        "contents": []
                    }
                    ]
                },
                "hero": {
                    "type": "image",
                    "url": "https://cdn.discordapp.com/attachments/1423258585621856372/1441321124007776256/Gemini_Generated_Image_myqyuvmyqyuvmyqy.png?ex=69215e66&is=69200ce6&hm=4354866e5bc51da016aea719daf5403fa6fb57c055c463207a1e848a00e4d670&",
                    "size": "full",
                    "aspectRatio": "1.51:1",
                    "aspectMode": "fit"
                },
                "body": {
                    "type": "box",
                    "layout": "vertical",
                    "contents": [
                        {
                            "type": "text",
                            "text": "點選[進入平台]，可快速進入",
                            "align": "start",
                            "contents": []
                        },{
                            "type": "text",
                            "text": "平台本體。",
                            "align": "start",
                            "contents": []
                        }
                    ]
                },
                "footer": {
                    "type": "box",
                    "layout": "horizontal",
                    "contents": [
                    {
                        "type": "button",
                        "action": {
                        "type": "uri",
                        "label": "點我觀看示意圖",
                        "uri": "https://cdn.discordapp.com/attachments/1423258585621856372/1441382801613131887/IMG_5784.png?ex=692197d7&is=69204657&hm=621cd650cedb24ef4004fec0d8b1ef5dad706a85cffdefaf74ed002919a85197&"
                        }
                    }
                    ]
                }
            }
        ]
    }
     reply=FlexSendMessage(
        alt_text="功能介紹",
        contents=flex_message
     )
     line_bot_api.reply_message(event.reply_token, reply)
     return
        
   
   """if text.lower() == "進入平台":
    reply = TemplateSendMessage(
        alt_text="this is a buttons template",   # 這裡用 alt_text，不是 altText
        template=ButtonsTemplate(
            thumbnail_image_url="https://cdn.discordapp.com/attachments/1423258585621856372/1423258680681566328/Gemini_Generated_Image_q3c1g8q3c1g8q3c1.jpg?ex=68dfa870&is=68de56f0&hm=ea04abaeaf2504bac49688f55a24b9bd73dbd401d8e62515fabb06d8bbf6f74e&",
            image_aspect_ratio="square",
            image_size="cover",
            image_background_color="#FFFFFF",
            title="校務窗口",
            text="請點選要進入的校務窗口",
            actions=[
                URIAction(
                    label="前往數位學園",
                    uri="https://elearning.nuu.edu.tw/mooc/index.php"
                ),
                URIAction(
                    label="前往校務資訊系統",
                    uri="https://eap10.nuu.edu.tw/Login.aspx?logintype=S"
                ),
                URIAction(
                    label="前往校內主頁",
                    uri="https://www.nuu.edu.tw/"
                )
            ]
        )
    )
    line_bot_api.reply_message(event.reply_token, reply)
    return"""
   
   #if text.lower()=="account":
      # 這裡定義了 flex_message
        #flex_message = create_flex_message()
        # ... (其他條件) ...


        # 函式結束時，您嘗試回覆訊息：
        #line_bot_api.reply_message(event.reply_token, [flex_message]) # <-- 錯誤發生在這裡！
        #return
     #如果 msg 不等於 "綁定"，則 flex_message 從未被賦值。
    #根據模型處理輸入
# 初始化 response 變數，以確保它總是有一個初始字串值

   # --- 模型處理區塊 (重構並強化防禦) ---
    
    # 1. 根據模型類型取得原始回應
   try:
        if model == "關鍵詞檢索":
            response= ask_rag_key(text)[0]
            #print(response)
        elif model == "資料結構化檢索":
            response = ask_sturag(text)[0]
            #print(response)
        elif model =="關鍵詞擴充檢索":
            response= ask_kgrag(text)[0]
            #print(response)
        elif model =="廣度檢索":
            response= ask_graphrag(text)[0]
            #print(response)
        elif model =="快速檢索":
            response= ask_cag(text)[0]
            print(response)
        elif model =="一般檢索":
            response= ask_rag(text)[0]
            #print(response)
        else:
            response = "尚未設定模型，請輸入方法更改來選擇。"
   except Exception as e:
        response = "處理失敗，請檢查 API 連線或模型設定。"
        
    # 2. 執行 Markdown 清理
   #print(response)
    # 3. 確保 response 永遠是非空字串
   MAX_LENGTH = 2000
   messages = []
    
   if not response:
        # 如果 clean_markdown 清理後為空，使用備用訊息
        if model not in ["一般檢索","資料結構化檢索", "關鍵詞檢索", "快速檢索","關鍵詞擴充檢索","廣度檢索"]:
            # 如果是模型未匹配導致的空，則發送模型提示
            response_text_to_send = "尚未設定模型，請輸入方法更改來選擇。"
        else:
            # 如果是模型生成空內容，則發送 AI 失敗提示
            response_text_to_send = "抱歉，AI未能根據您的輸入產生有效且可見的文本回覆。請嘗試不同問題。"
   else:
        response_text_to_send=clean_markdown_format(response)
    
    # 4. 訊息分割邏輯
   current_index = 0
   while current_index < len(response_text_to_send):
        # 擷取 2000 字元
        chunk = response_text_to_send[current_index:current_index + MAX_LENGTH]
        messages.append(TextSendMessage(text=chunk))
        current_index += MAX_LENGTH

    # 5. 避免 messages 列表為空 (最終安全網)
   if not messages:
        messages.append(TextSendMessage(text="發生未知錯誤，無法傳送訊息。"))

    # 6. 回覆結果
    # Line 449 (修正後的位置)
   line_bot_api.reply_message(event.reply_token, messages)
   pure_text = convert_to_text(messages)
   last_reply[user_id] = pure_text
    







