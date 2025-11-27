const csrfToken = window.csrfToken;
// 1. 先宣告 currentChatId，預設 null
let currentChatId = null;
// 2. DOMContentLoaded 時，從 URL 讀 chat_history_id，設定 currentChatId
document.addEventListener("DOMContentLoaded", () => {
  const params = new URLSearchParams(window.location.search);
  if (params.has('chat_history_id')) {
    currentChatId = params.get('chat_history_id');
    console.log("已選擇歷史對話，ID =", currentChatId);
  }
  refreshChatHistorySidebar();

  setTimeout(() => {
    document.querySelectorAll(".message").forEach(div => {
      let content = div.innerHTML.trim();

      content = content.replace(/(<\/[a-z]+>)\s*(\n\s*){2,}(<\w+>)/gi, '$1\n$3');

      if (content) {
        div.innerHTML = marked.parseInline(content);
        if (window.MathJax) {
          MathJax.typesetPromise([div]);
        }
      }
    });
    scrollToBottom();
  }, 100);
});

function loadChatHistory(chatId) {
    currentChatId = chatId;
    console.log("正在加載歷史對話，ID =", currentChatId);
    window.location.href = `?chat_history_id=${currentChatId}`;
}

function toggleModelMenu() {
    const menu = document.getElementById("model-popup-menu");
    menu.style.display = menu.style.display === "block" ? "none" : "block";
}

// 點擊其他地方時自動關閉選單
document.addEventListener("click", function (event) {
    const btn = document.querySelector(".model-toggle-btn");
    const menu = document.getElementById("model-popup-menu");
    if (!btn.contains(event.target) && !menu.contains(event.target)) {
        menu.style.display = "none";
    }
});

// 1. 監聽 textarea 輸入框的 keydown 事件
document.getElementById("message").addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();  // 阻止換行
        sendMessage();           // 呼叫發送訊息函式
    }
});

function scrollToBottom() {
    const mainContent = document.querySelector(".main-content-m");
    mainContent.scrollTo({ top: mainContent.scrollHeight, behavior: 'smooth' });
}

const renderBotMessage = async (html, modelName = "AI Model", links = [], imageLinks = [], duration) => {
    const container = document.getElementById("messages");

    // ===== 外層 wrapper =====
    const wrapper = document.createElement("div");
    wrapper.classList.add("message-container", "bot-message-container");

    // ===== message-header =====
    const messageHeader = document.createElement("div");
    messageHeader.classList.add("message-header");

    // 模型名稱
    const modelBox = document.createElement("div");
    modelBox.classList.add("model-name-box");
    const modelText = document.createElement("div");
    modelText.classList.add("model-name-text");
    const displayNameMap = {
        "auto": "自動模式",
        rag: "一般檢索",
        cag: "快速檢索",
        sturag: "資料結構化檢索",
        rag_key: "關鍵詞檢索",
        kgrag: "關鍵詞擴充檢索",
        graphrag: "廣度檢索",
        direct: "一般回應"

    };
    modelText.textContent = displayNameMap[modelName] || modelName;
    modelBox.appendChild(modelText);
    messageHeader.appendChild(modelBox);

    // ===== 相關連結（放在 header） =====
    if (links && links.length > 0) {
        const linksWrapper = document.createElement('div');
        linksWrapper.classList.add('related-links-wrapper');

        const linksButton = document.createElement('button');
        linksButton.className = 'related-links-icon-btn';
        linksButton.innerHTML = `<i class="bi bi-link-45deg"></i>`;
        linksButton.onclick = () => toggleFloatingLinks(linksButton);

        const floatingLinksContainer = document.createElement('div');
        floatingLinksContainer.className = 'floating-links-container';

        const linksList = document.createElement('div');
        linksList.classList.add('links-list');

        links.forEach(link => {
            if (link && link.url && link.title) {
                const a = document.createElement('a');
                a.href = link.url;
                a.textContent = link.title;
                a.target = "_blank";
                a.className = "related-link-item";
                linksList.appendChild(a);
            }
        });

        floatingLinksContainer.appendChild(linksList);
        linksWrapper.appendChild(linksButton);
        linksWrapper.appendChild(floatingLinksContainer);
        messageHeader.appendChild(linksWrapper);
    }

    // header 加到 wrapper
    wrapper.appendChild(messageHeader);

    // ===== message 內容 =====
    const message = document.createElement("div");
    message.classList.add("message", "bot-message");
    message.innerHTML = marked.parseInline(html.trim());
    wrapper.appendChild(message);

    // ===== 圖片 =====
    if (imageLinks && imageLinks.length > 0) {
        const imageGallery = document.createElement('div');
        imageGallery.classList.add('image-gallery');
        wrapper.appendChild(imageGallery);

        imageLinks.forEach(link => {
            const a = document.createElement('a');
            a.href = link;
            a.target = "_blank";

            const img = document.createElement('img');
            img.src = link;
            img.alt = "來源圖片";
            img.onload = scrollToBottom;
            img.onerror = () => console.error("圖片載入失敗:", link);

            a.appendChild(img);
            imageGallery.appendChild(a);
        });
    }

    // ===== 耗時 =====
    if (duration != null) {
        const durationEl = document.createElement("div");
        durationEl.classList.add("message-duration");
        durationEl.textContent = `（耗時：${duration.toFixed(2)}秒）`;
        wrapper.appendChild(durationEl);
    }

    // ===== MathJax 處理 =====
    if (window.MathJax) await MathJax.typesetPromise([message]);

    container.appendChild(wrapper);
    scrollToBottom();
};

function toggleFloatingLinks(button) {
    const wrapper = button.closest('.related-links-wrapper');
    const linksContainer = wrapper.querySelector('.floating-links-container');
    
    document.querySelectorAll('.floating-links-container.show').forEach(c => {
        if (c !== linksContainer) {
            c.classList.remove('show');
        }
    });
    linksContainer.classList.toggle('show');
}

document.addEventListener('click', function(event) {
    const isClickInsideLinkButton = event.target.closest('.related-links-icon-btn');
    const isClickInsideContainer = event.target.closest('.floating-links-container');

    if (!isClickInsideLinkButton && !isClickInsideContainer) {
        document.querySelectorAll('.floating-links-container.show').forEach(container => {
            container.classList.remove('show');
        });
    }
});

async function sendMessage() {
    const input = document.getElementById("message");
    const raw = input.value.trim();
    if (!raw) return;

    const messagesDiv = document.getElementById("messages");

    // 使用者訊息
    const userMessageContainer = document.createElement("div");
    userMessageContainer.classList.add("message-container", "user-message-container");
    const userMessage = document.createElement("div");
    userMessage.classList.add("message", "user-message");
    userMessage.innerHTML = marked.parseInline(raw);
    userMessageContainer.appendChild(userMessage);
    messagesDiv.appendChild(userMessageContainer);

    // Spinner
    const loadingMessage = document.createElement("div");
    loadingMessage.classList.add("message-container", "bot-message-container");
    const spinner = document.createElement("div");
    spinner.classList.add("loading-spinner");
    loadingMessage.appendChild(spinner);
    messagesDiv.appendChild(loadingMessage);

    scrollToBottom();

    const selectedModel = document.querySelector('input[name="model"]:checked').value;
    const modelToSend = selectedModel === 'auto' ? 'auto-detect' : selectedModel;

    try {
    const response = await fetch('/chat_api/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken
        },
        body: JSON.stringify({
            message: raw,
            chat_history_id: currentChatId,
            model: modelToSend
        })
    });

    const data = await response.json();
    currentChatId = data.chat_history_id;
    refreshChatHistorySidebar();

    if (messagesDiv.contains(loadingMessage)) {
        messagesDiv.removeChild(loadingMessage);
    }

    const botResponseContent = data.response || "";
    const relatedLinks = data.related_links || [];
    const imageLinks = data.image_links || [];
    const duration = data.duration;
    const usedModel = data.model || selectedModel;

    const imageExtensions = ['.png', '.jpg', '.jpeg', '.gif', '.webp']; 

    relatedLinks.forEach(link => {
        if (link && typeof link.url === 'string') {
            const isImage = imageExtensions.some(ext => link.url.toLowerCase().endsWith(ext));
            if (isImage) {
                imageLinks.push(link.url); 
            }
        }
    });

    await renderBotMessage(botResponseContent, usedModel, relatedLinks, imageLinks, duration);

} catch (error) {
    console.error("Error:", error);
    if (messagesDiv.contains(loadingMessage)) {
        messagesDiv.removeChild(loadingMessage);
    }
    const errorMsgContainer = document.createElement("div");
    errorMsgContainer.classList.add("message-container", "bot-message-container");
    const errorMsg = document.createElement("div");
    errorMsg.classList.add("message", "bot-message");
    errorMsg.textContent = "出現錯誤，請稍後再試。";
    errorMsgContainer.appendChild(errorMsg);
    messagesDiv.appendChild(errorMsgContainer);
    scrollToBottom();
}


    input.value = "";
    if (window.MathJax) {
        await MathJax.typesetPromise([userMessage]);
    }
    scrollToBottom();
}

document.getElementById('login-banner')?.addEventListener('click', () => {
    document.getElementById('login-modal').style.display = 'block';
});

document.getElementById('login-form').addEventListener('submit', function(e) {
    e.preventDefault();

    const email = document.getElementById('email').value.trim();
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value.trim();
    const verification_code = document.getElementById('verification_code').value.trim();

    fetch('/login_api/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken },
        body: JSON.stringify({ 
            email,
            username,
            password,
            verification_code
        })
    })
    .then(res => res.json())
    .then(data => {
        alert(data.message);
        if (data.success) {
            location.reload();
        }
    })
    .catch(() => {
        alert('登入時發生錯誤');
    });
});


// 新增發送驗證碼的事件監聽
document.getElementById('send-code-btn').addEventListener('click', function() {
  const email = document.getElementById('email').value.trim();
  if (!email) {
    alert('請先輸入電子郵件！');
    return;
  }

  // 將發送驗證碼的請求導向 /login_api/
  fetch('/login_api/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken },
      body: JSON.stringify({ email: email })
  })
  .then(res => res.json())
  .then(data => {
      alert(data.message);
  })
  .catch(() => {
      alert('發送驗證碼時發生錯誤');
  });
});

function closeLoginModal() {
    document.getElementById('login-modal').style.display = 'none';
}
window.addEventListener('click', e => {
    const modal = document.getElementById('login-modal');
    if (e.target === modal) modal.style.display = 'none';
});
function goToSettings() {
    window.location.href = window.settingsUrl;
}
function goToCalendar() {
    window.location.href = window.calendarUrl;
}
function goToGuide() {
    window.location.href = window.guideUrl;
}
window.toggleMenu = function () {
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.querySelector('.main-content-m');
    const loginBanner = document.getElementById('login-banner');
    const loginBanner2 = document.getElementById('login-banner-2');
    const inputArea = document.querySelector('.input-area-m'); 

    sidebar.classList.toggle('show');
    mainContent.classList.toggle('show');
    if (sidebar.classList.contains('show')) {
        inputArea.style.left = '300px'; 
        inputArea.style.width = 'calc(100% - 250px)'; 
        inputArea.style.transform = 'none'; 
    } else {
        inputArea.style.left = '50%';
        inputArea.style.transform = 'translateX(-50%)';
        inputArea.style.width = '100%'; 
    }
    if (loginBanner) {
        if (sidebar.classList.contains('show')) {
            loginBanner.style.left = '68px';
            loginBanner2.style.left = '37px';
        } else {
            loginBanner.style.left = '-250px';
            loginBanner2.style.left = '-400px';
        }
    }
}
const light = document.getElementById('cursor-light');
document.addEventListener('mousemove', e => {
    light.style.left = e.clientX + 'px';
    light.style.top  = e.clientY + 'px';
});

console.log("目前登入帳號：", window.userEmail);
function createNewChatHistory() {
    window.location.href = window.location.pathname;
}

function addMessage(rawMd, username) {
    const container = document.getElementById("messages");
    const html = marked.parse(rawMd);
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message");
    msgDiv.classList.add(username === "bot" ? "bot-message" : "user-message");
    msgDiv.innerHTML = html;
    container.appendChild(msgDiv);
    if (window.MathJax) {
        MathJax.typesetPromise([msgDiv]);
    }
}

function refreshChatHistorySidebar() {
    fetch('/chat_history_api/')
        .then(response => response.json())
        .then(data => {
            const sidebarContainer = document.querySelector(".chat-history");
            sidebarContainer.innerHTML = '';

            data.forEach(chat => {
                const chatDiv = document.createElement("div");
                chatDiv.classList.add("chat-entry");

                // 連結
                const link = document.createElement("a");
                link.href = `?chat_history_id=${chat.id}`;
                link.setAttribute("data-chat-id", chat.id);
                link.onclick = () => loadChatHistory(chat.id);
                link.textContent = chat.title;

                const deleteBtn = document.createElement("button");
                deleteBtn.innerHTML = '<i class="bi bi-trash"></i>';
                deleteBtn.classList.add("delete-chat-btn");
                deleteBtn.onclick = (event) => {
                    event.stopPropagation(); 
                    deleteChatHistory(chat.id);
                };

                chatDiv.appendChild(link);
                chatDiv.appendChild(deleteBtn);
                sidebarContainer.appendChild(chatDiv);
            });
        })
        .catch(error => {
            console.error("無法更新側邊欄聊天紀錄：", error);
        });
}

function loadChatHistory(chatId) {
    currentChatId = chatId;
    console.log("正在加載歷史對話，ID =", currentChatId);
    window.location.href = `?chat_history_id=${currentChatId}`;
}

function deleteChatHistory(chatId) {
    if (!confirm("確定要刪除此對話紀錄嗎？")) return;

    fetch(`/chat_history_api/${chatId}/`, {
        method: 'DELETE',
        headers: {
            'X-CSRFToken': csrfToken
        }
    })
    .then(response => {
        if (response.ok) {
            refreshChatHistorySidebar();
        } else {
            alert("刪除失敗");
        }
    })
    .catch(error => {
                console.error("刪除失敗：", error);
    });
}

window.submitFeedback = function () {
    const content = document.getElementById("feedback-content").value.trim();
    const responseEl = document.getElementById("feedback-response");

    if (!content) {
        responseEl.textContent = "請填寫回饋內容！";
        responseEl.style.color = "orange";
        return;
    }

    fetch(window.submitFeedbackUrl, {  
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCookie('csrftoken')
        },
        body: JSON.stringify({ content: content })
    })
    .then(res => res.json())
    .then(data => {
        responseEl.textContent = data.message;
        responseEl.style.color = data.success ? "green" : "red";
        if (data.success) {
            document.getElementById("feedback-content").value = "";
        }
    })
    .catch(err => {
        responseEl.textContent = "提交失敗：" + err;
        responseEl.style.color = "red";
    });
};

// 在 feedbackModal 顯示時，動態建立並綁定按鈕
const feedbackModalEl = document.getElementById('feedbackModal');
feedbackModalEl.addEventListener('show.bs.modal', function () {
    const actionsContainer = document.getElementById('feedbackModalActions');

    // 動態建立所有外部按鈕
    actionsContainer.innerHTML = `
        <button type="button" class="modal-action-btn modal-close-outside" title="關閉" aria-label="關閉" data-bs-dismiss="modal">
            <i class="bi bi-x-lg"></i>
        </button>
        <button id="submit-feedback-btn" class="modal-action-btn" title="送出回饋" aria-label="送出回饋">
            <i class="bi bi-send"></i>
        </button>
    `;
    actionsContainer.setAttribute('aria-hidden', 'false');

    // 綁定「送出」按鈕的事件
    document.getElementById('submit-feedback-btn').addEventListener('click', () => {
        submitFeedback();
    });
});

// 當 feedbackModal 隱藏時清空外部按鈕容器，避免殘留事件綁定
feedbackModalEl.addEventListener('hidden.bs.modal', function () {
    const actions = document.getElementById('feedbackModalActions');
    if (actions) {
        actions.innerHTML = '';
        actions.setAttribute('aria-hidden', 'true');
    }
});

function openImageModal(src) {
    const modal = document.getElementById("image-modal");
    const modalImg = document.getElementById("image-modal-img");
    modal.style.display = "flex"; // 顯示模態視窗
    modalImg.src = src; // 設定圖片來源
}

function closeImageModal() {
    const modal = document.getElementById("image-modal");
    modal.style.display = "none"; // 隱藏模態視窗
}

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

(function () {
  const textarea = document.getElementById('message');
  const voiceBtn  = document.getElementById('voice-btn');
  if (!voiceBtn) return;

  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

  // 瀏覽器不支援時停用按鈕
  if (!SR) {
    voiceBtn.disabled = true;
    voiceBtn.title = '此瀏覽器不支援語音輸入';
    return;
  }

  const recognition = new SR();
  recognition.lang = 'zh-TW';         // 中文（台灣）
  recognition.interimResults = true;  // 啟用暫時結果
  recognition.continuous = false;     // 每次一句話

  let listening = false;

  function startListening() {
    try { recognition.start(); } catch (e) {}
  }
  function stopListening() {
    try { recognition.stop(); } catch (e) {}
  }

  // 開始錄音 UI
  recognition.onstart = () => {
    listening = true;
    voiceBtn.classList.add('listening');
    voiceBtn.setAttribute('aria-pressed', 'true');
    textarea.dataset.finalText = textarea.value; // 保存目前內容
  };

  // 停止錄音 UI
  recognition.onend = () => {
    listening = false;
    voiceBtn.classList.remove('listening');
    voiceBtn.setAttribute('aria-pressed', 'false');
    // 錄音結束時，保留已完成文字
    textarea.value = textarea.dataset.finalText || textarea.value;
    delete textarea.dataset.finalText;
  };

  // 即時顯示 + 最終結果
  recognition.onresult = (event) => {
    let finalText = textarea.dataset.finalText || '';
    let interimText = '';

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalText += t;
      } else {
        interimText += t;
      }
    }

    // 保存已完成部分
    textarea.dataset.finalText = finalText;

    // 即時更新輸入框
    textarea.value = finalText + interimText;

    // 觸發輸入框變動事件（若有自動縮高功能）
    textarea.dispatchEvent(new Event('input'));
  };

  recognition.onerror = (e) => {
    console.warn('Speech error:', e.error);
  };

  // 點擊語音鍵啟停
  voiceBtn.addEventListener('click', () => {
    if (listening) stopListening();
    else startListening();
  });

  // 空白鍵啟停（焦點不在輸入框時）
  document.addEventListener('keydown', (ev) => {
    if (ev.code === 'Space' && document.activeElement !== textarea) {
      ev.preventDefault();
      if (listening) stopListening();
      else startListening();
    }
  });
})();