const calendarTitle = document.getElementById("calendar-title");
const calendarBody = document.getElementById("calendar-body");
const prevBtn = document.getElementById("prev-month");
const nextBtn = document.getElementById("next-month");

let dbEvents = [];
let userEvents = [];
let pinnedEvents = []; // 新增一個變數來儲存已釘選的事件列表

let today = new Date();
let currentYear = today.getFullYear();
let currentMonth = today.getMonth();

function loadEventsForMonth(year, month) {
    const dbPromise = fetch(`/events/?year=${year}&month=${month + 1}`)
        .then(response => response.json())
        .catch(error => {
            console.error("載入資料庫事件失敗", error);
            return [];
        });

    const userPromise = fetch(`/my_events/?year=${year}&month=${month + 1}`)
        .then(response => response.json())
        .catch(error => {
            console.error("載入使用者事件失敗", error);
            return [];
        });
        
    const pinnedPromise = fetch("/get_pinned_events/")
        .then(res => res.json())
        .catch(error => {
            console.error("載入已釘選事件失敗", error);
            return [];
        });

    Promise.all([dbPromise, userPromise, pinnedPromise])
        .then(([dbData, userData, pinnedData]) => {
            dbEvents = dbData;
            userEvents = userData;
            pinnedEvents = pinnedData;
            renderCalendar(currentYear, currentMonth);
        })
        .catch(error => {
            console.error("載入所有事件失敗", error);
            renderCalendar(currentYear, currentMonth);
        });
}

function renderCalendar(year, month) {
    calendarBody.innerHTML = "";
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const startWeekday = firstDay.getDay();
    const totalDays = lastDay.getDate();
    let row = document.createElement("tr");

    for (let i = 0; i < startWeekday; i++) {
        row.innerHTML += "<td></td>";
    }

    const allEvents = [...dbEvents, ...userEvents];

    for (let day = 1; day <= totalDays; day++) {
        const date = new Date(year, month, day);
        const dateStr = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
        const cell = document.createElement("td");
        const isWeekend = date.getDay() === 0 || date.getDay() === 6;
        const isToday = date.toDateString() === new Date().toDateString();

        const cellDiv = document.createElement("div");
        cellDiv.innerHTML = `${day}<br><small>---</small>`;
        cell.appendChild(cellDiv);

        if (isWeekend) cell.classList.add("weekend");
        if (isToday) cell.classList.add("today");

        cell.addEventListener("click", () => {
            showCreateModal(dateStr);
        });

        const eventsForDay = allEvents.filter(e => e.date === dateStr);
        const placeholder = cellDiv.querySelector("small");
        if (eventsForDay.length > 0 && placeholder) {
            placeholder.remove();
        }

        eventsForDay.forEach(event => {
            const eventElem = document.createElement("div");
            const displayTitle = event.title.length > 42 ? event.title.substring(0, 42) + "..." : event.title;
            eventElem.textContent = displayTitle; 
            eventElem.classList.add("custom-event");

            // 檢查是否已釘選
            const isPinned = pinnedEvents.some(p => {
                const eventId = event.is_user_event ? `user_event_${event.id}` : event.id;
                return p.id === eventId;
            });
            
            eventElem.dataset.isPinned = isPinned;

            if (event.is_user_event) {
                eventElem.classList.add("user-event-style");
            } else {
                eventElem.classList.add("db-event-style");
            }
            
            eventElem.addEventListener("click", (e) => {
                e.stopPropagation();
                if (event.is_user_event) {
                    showUserEventModal(event);
                } else {
                    showEventModal(event);
                }
            });

            cell.appendChild(eventElem);
        });

        row.appendChild(cell);

        if ((startWeekday + day) % 7 === 0 || day === totalDays) {
            calendarBody.appendChild(row);
            row = document.createElement("tr");
        }
    }
    calendarTitle.textContent = `${year - 1911}年${month + 1}月`;
}

function showCreateModal(dateStr) {
    const createEventDateInput = document.getElementById("createEventDate");
    const actionsContainer = document.getElementById("createModalActions");

    if (createEventDateInput) {
        createEventDateInput.value = dateStr;
        
        // 動態建立所有外部按鈕
        actionsContainer.innerHTML = `
            <button type="button" class="modal-action-btn modal-close-outside" title="關閉" aria-label="關閉" data-bs-dismiss="modal">
                <i class="bi bi-x-lg"></i>
            </button>
            <button id="add-event-btn" class="modal-action-btn" title="新增事件" aria-label="新增事件">
                <i class="bi bi-plus-lg"></i>
            </button>
        `;
        actionsContainer.setAttribute('aria-hidden', 'false');

        // 綁定「新增」按鈕的事件
        document.getElementById('add-event-btn').addEventListener('click', () => {
            // 觸發表單的 submit 事件
            document.getElementById('create-event-form').dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
        });

        // 顯示彈窗
        const modal = new bootstrap.Modal(document.getElementById("createEventModal"));
        modal.show();
    } else {
        console.warn("createEventModal not found.");
    }
}

document.getElementById("create-event-form").addEventListener("submit", function(e) {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);
    fetch('/create_event/', {
        method: 'POST',
        body: formData,
        headers: {
            "X-CSRFToken": getCookie('csrftoken')
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert("事件新增成功！");
            bootstrap.Modal.getInstance(document.getElementById("createEventModal")).hide();
            loadEventsForMonth(currentYear, currentMonth);
            refreshUserEventList();
        } else {
            alert("事件新增失敗：" + data.message);
        }
    })
    .catch(error => console.error('新增事件失敗:', error));
});

// 處理使用者自訂事件的彈窗
function showUserEventModal(event) {
  const modalTitle = document.getElementById("eventModalLabel");
  const modalBody = document.getElementById("eventModalBody");
  const actionsContainer = document.getElementById("eventModalActions");

  const isPinned = pinnedEvents.some(p => p.id === `user_event_${event.id}`);

  modalTitle.textContent = event.title;
  modalBody.innerHTML = `
    <p><strong>日期：</strong>${event.date}</p>
    <p><strong>內容：</strong>${event.description || "（無詳細說明）"}</p>
  `;

  // 建立外部按鈕：最上方為 x（關閉），下方垂直放釘選/修改/刪除
  actionsContainer.innerHTML = `
    <button type="button" class="modal-action-btn modal-close-outside" id="modal-close-outside"
            title="關閉" aria-label="關閉" data-bs-dismiss="modal">
      <i class="bi bi-x-lg"></i>
    </button>
    <button id="pin-btn" class="modal-action-btn icon-btn" aria-label="釘選按鈕">
      <i id="pin-icon" class="bi"></i>
    </button>
    <button id="edit-user-btn" class="modal-action-btn" title="修改" aria-label="修改">
      <i class="bi bi-pencil"></i>
    </button>

    <button id="delete-user-btn" class="modal-action-btn" title="刪除" aria-label="刪除">
      <i class="bi bi-trash"></i>
    </button>
  `;
  actionsContainer.setAttribute('aria-hidden', 'false');

  // 綁定按鈕事件（釘選）
  const pinBtn = document.getElementById('pin-btn');
  const pinIcon = document.getElementById('pin-icon');
  let pinned = isPinned;
  function updatePinButtonState() {
    if (pinned) {
      pinIcon.classList.remove('bi-pin-angle');
      pinIcon.classList.add('bi-pin-angle-fill');
      pinBtn.classList.add('is-pinned');
    } else {
      pinIcon.classList.remove('bi-pin-angle-fill');
      pinIcon.classList.add('bi-pin-angle');
      pinBtn.classList.remove('is-pinned');
    }
  }
  updatePinButtonState();

  pinBtn.addEventListener('click', () => {
    const eventData = {
      id: event.id,
      title: event.title,
      date: event.date,
      description: event.description,
      is_user_event: true,
      url: ""
    };
    pinEvent(eventData)
      .then(data => {
        if (data.success) {
          pinned = !pinned;
          updatePinButtonState();
          loadEventsForMonth(currentYear, currentMonth);
          refreshMainEventList();
        } else {
          console.error("釘選操作失敗:", data.message);
        }
      })
      .catch(error => { console.error("釘選操作失敗", error); });
  });

  // 修改按鈕
  document.getElementById('edit-user-btn').addEventListener('click', () => {
    showEditUserEventModal(event.id);
  });

  // 刪除按鈕
  document.getElementById('delete-user-btn').addEventListener('click', () => {
    deleteUserEvent(event.id);
  });

  // (可選) 如果你需要在按 x 時做其他清理，監聽 close 按鈕：
  document.getElementById('modal-close-outside').addEventListener('click', () => {
    // 這裡可以放按下 x 時要做的事（大多數情況不需要，Bootstrap 會自動關閉 modal）
  });

  const modal = new bootstrap.Modal(document.getElementById("eventModal"));
  modal.show();
}



function showEditUserEventModal(eventId) {
    const eventToEdit = userEvents.find(e => e.id === eventId);
    if (!eventToEdit) {
        alert("找不到此事件。");
        return;
    }

    // 先關閉舊的彈窗
    const eventModalInstance = bootstrap.Modal.getInstance(document.getElementById("eventModal"));
    if (eventModalInstance) {
        eventModalInstance.hide();
    }

    const actionsContainer = document.getElementById("editModalActions");
    
    // 1. 動態建立**所有**外部按鈕，垂直排列
    actionsContainer.innerHTML = `
        <button type="button" class="modal-action-btn modal-close-outside" title="關閉" aria-label="關閉" data-bs-dismiss="modal">
            <i class="bi bi-x-lg"></i>
        </button>
        <button id="save-edit-btn" class="modal-action-btn" title="儲存變更" aria-label="儲存變更">
            <i class="bi bi-save"></i>
        </button>
        <button id="delete-user-btn" class="modal-action-btn" title="刪除" aria-label="刪除">
            <i class="bi bi-trash"></i>
        </button>
    `;
    actionsContainer.setAttribute('aria-hidden', 'false');

    // 2. 填充表單並顯示彈窗
    document.getElementById('editEventId').value = eventToEdit.id;
    document.getElementById('editEventTitle').value = eventToEdit.title;
    document.getElementById('editEventDate').value = eventToEdit.date;
    document.getElementById('editEventDescription').value = eventToEdit.description;
    
    // 3. 綁定按鈕事件
    // 儲存變更按鈕的事件
    document.getElementById('save-edit-btn').addEventListener('click', () => {
        // 觸發表單的 submit 事件
        document.getElementById('edit-event-form').dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
    });
    
    // 刪除按鈕的事件
    document.getElementById('delete-user-btn').addEventListener('click', () => {
        deleteUserEvent(eventId);
    });

    // 4. 創建並顯示彈窗
    const editModal = new bootstrap.Modal(document.getElementById("editEventModal"));
    editModal.show();
}

document.getElementById('edit-event-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const eventId = document.getElementById('editEventId').value;
    const formData = new FormData(e.target);

    fetch(`/update_event/${eventId}/`, {
        method: 'POST',
        body: formData,
        headers: {
            "X-CSRFToken": getCookie('csrftoken')
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert("事件修改成功！");
            bootstrap.Modal.getInstance(document.getElementById("editEventModal")).hide();
            loadEventsForMonth(currentYear, currentMonth);
            refreshUserEventList();
            refreshMainEventList();
        } else {
            alert("事件修改失敗：" + data.message);
        }
    })
    .catch(error => console.error('修改事件失敗:', error));
});

function deleteUserEvent(eventId) {
    if (!confirm("確定要刪除此事件嗎？")) {
        return;
    }

    fetch(`/delete_event/${eventId}/`, {
        method: 'POST',
        headers: {
            "X-CSRFToken": getCookie('csrftoken')
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert("事件刪除成功！");
            const eventModalEl = document.getElementById("eventModal");
            const modalInstance = bootstrap.Modal.getInstance(eventModalEl);
            if (modalInstance) {
                modalInstance.hide();
            }
            loadEventsForMonth(currentYear, currentMonth);
            refreshUserEventList();
            refreshMainEventList();
        } else {
            alert("事件刪除失敗：" + data.message);
        }
    })
    .catch(error => console.error('刪除事件失敗:', error));
}


function refreshUserEventList() {
    fetch("/get_user_events/")
        .then(res => res.json())
        .then(data => {
            const list = document.getElementById("user-event-list");
            list.innerHTML = "";
            if (data.length === 0) {
                const noEventsDiv = document.createElement("div");
                noEventsDiv.className = "chat-entry";
                noEventsDiv.textContent = "沒有自訂事件";
                list.appendChild(noEventsDiv);
            } else {
                data.forEach(e => {
                    const entry = document.createElement("div");
                    entry.className = "chat-entry";
                    const eventData = JSON.stringify({
                        id: e.id,
                        title: e.title,
                        date: e.date,
                        description: e.description,
                        is_user_event: true
                    }).replace(/"/g, '&quot;');
                    entry.innerHTML = `
                        <div class="d-flex justify-content-between align-items-center w-100">
                            <a href="#" class="event-entry-link flex-grow-1" onclick="showPinnedEventModal(${eventData})">${e.title}</a>
                            <button class="reminder-btn" onclick="showSetReminderModal(${eventData})" title="設定提醒">
                                <i class="bi bi-bell"></i>
                            </button>
                        </div>
                    `;
                    list.appendChild(entry);
                });
            }
        })
        .catch(error => {
            console.error("載入使用者事件失敗", error);
            const list = document.getElementById("user-event-list");
            list.innerHTML = '<div class="chat-entry">載入使用者事件失敗</div>';
        });
}

document.addEventListener("DOMContentLoaded", () => {
    refreshMainEventList();
    refreshUserEventList();
});

function showEventModal(event) {
  const modalTitle = document.getElementById("eventModalLabel");
  const modalBody = document.getElementById("eventModalBody");
  const actionsContainer = document.getElementById("eventModalActions");

  modalTitle.textContent = event.title;
  modalBody.innerHTML = `
    <p><strong>日期：</strong>${event.date}</p>
    <p><strong>連結：</strong><a href="${event.url}" target="_blank">點我</a></p>
    <p><strong>內容：</strong>${event.description || "（無詳細說明）"}</p>
  `;

  // 只有釘選 + x
  actionsContainer.innerHTML = `
    <button type="button" class="modal-action-btn modal-close-outside" id="modal-close-outside"
            title="關閉" aria-label="關閉" data-bs-dismiss="modal">
      <i class="bi bi-x-lg"></i>
    </button>
    <button id="pin-btn" class="modal-action-btn icon-btn" aria-label="釘選按鈕">
      <i id="pin-icon" class="bi"></i>
    </button>
  `;
  actionsContainer.setAttribute('aria-hidden', 'false');

  const pinBtn = document.getElementById('pin-btn');
  const pinIcon = document.getElementById('pin-icon');

  let pinned = event.is_pinned;
  function updatePinButtonState() {
    if (pinned) {
      pinIcon.classList.remove('bi-pin-angle');
      pinIcon.classList.add('bi-pin-angle-fill');
      pinBtn.classList.add('is-pinned');
    } else {
      pinIcon.classList.remove('bi-pin-angle-fill');
      pinIcon.classList.add('bi-pin-angle');
      pinBtn.classList.remove('is-pinned');
    }
  }
  updatePinButtonState();

  pinBtn.addEventListener('click', () => {
    const eventData = {
      id: event.id,
      title: event.title,
      date: event.date,
      url: event.url,
      description: event.description,
      is_user_event: false
    };
    pinEvent(eventData)
      .then(data => {
        if (data.success) {
          pinned = !pinned;
          updatePinButtonState();
          loadEventsForMonth(currentYear, currentMonth);
          refreshMainEventList();
        } else {
          console.error("釘選操作失敗:", data.message);
        }
      })
      .catch(error => { console.error("釘選操作失敗", error); });
  });

  const modal = new bootstrap.Modal(document.getElementById("eventModal"));
  modal.show();
}



function pinEvent(eventData) {
    const formData = new URLSearchParams();
    formData.append('id', eventData.id);
    formData.append('title', eventData.title);
    formData.append('date', eventData.date);
    formData.append('url', eventData.url);
    formData.append('description', eventData.description);
    formData.append('is_user_event', eventData.is_user_event);
    
    return fetch("/pin_event/", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            "X-CSRFToken": getCookie('csrftoken')
        },
        body: formData
    })
    .then(res => res.json());
}

function showEventModalFromSidebar(eventData) {
    showEventModal({
        id: eventData.id,
        title: eventData.title,
        date: eventData.date,
        url: eventData.url,
        description: eventData.description,
        is_pinned: true
    });
}

function refreshMainEventList() {
    fetch("/get_pinned_events/")
        .then(res => res.json())
        .then(data => {
            const list = document.getElementById("main-event-list");
            list.innerHTML = "";
            if (data.length === 0) {
                const noEventsDiv = document.createElement("div");
                noEventsDiv.className = "chat-entry";
                noEventsDiv.textContent = "沒有已釘選的事件";
                list.appendChild(noEventsDiv);
            } else {
                data.forEach(e => {
                    const entry = document.createElement("div");
                    entry.className = "chat-entry";
                    const eventData = JSON.stringify({
                        id: e.id,
                        title: e.title,
                        date: e.date,
                        url: e.url,
                        description: e.description,
                        is_user_event: e.is_user_event, // 新增這個欄位
                        is_pinned: true
                    }).replace(/"/g, '&quot;');
                    
                    // entry.innerHTML = `<a href="#" class="event-entry-link" onclick="showPinnedEventModal(${eventData})">${e.title}</a>`;
                    // --- 新增提醒按鈕 ---
                    entry.innerHTML = `
                        <div class="d-flex justify-content-between align-items-center w-100">
                            <a href="#" class="event-entry-link flex-grow-1" onclick="showPinnedEventModal(${eventData})">${e.title}</a>
                            <button class="reminder-btn" onclick="showSetReminderModal(${eventData})" title="設定提醒">
                                <i class="bi bi-bell"></i>
                            </button>
                        </div>
                    `;
                    list.appendChild(entry);
                });
            }
        })
        .catch(error => {
            console.error("載入已釘選事件失敗", error);
            const list = document.getElementById("main-event-list");
            list.innerHTML = '<div class="chat-entry">載入已釘選事件失敗</div>';
        });
}

function showPinnedEventModal(eventData) {
    if (eventData.is_user_event) {
        // 處理使用者自訂事件
        const originalEventId = parseInt(eventData.id.replace('user_event_', ''));
        const originalEvent = userEvents.find(e => e.id === originalEventId);
        if (originalEvent) {
            // 為了顯示修改/刪除按鈕，重新呼叫 showUserEventModal
            showUserEventModal(originalEvent);
        } else {
            alert("找不到此自訂事件");
        }
    } else {
        // 處理外部文章事件
        showEventModal(eventData);
    }
}


prevBtn.addEventListener("click", () => {
    currentMonth--;
    if (currentMonth < 0) {
        currentMonth = 11;
        currentYear--;
    }
    loadEventsForMonth(currentYear, currentMonth);
});
nextBtn.addEventListener("click", () => {
    currentMonth++;
    if (currentMonth > 11) {
        currentMonth = 0;
        currentYear++;
    }
    loadEventsForMonth(currentYear, currentMonth);
});
loadEventsForMonth(currentYear, currentMonth);

// 當 eventModal 隱藏時清空外部按鈕內容，避免殘留事件綁定
const eventModalEl = document.getElementById('eventModal');
eventModalEl.addEventListener('hidden.bs.modal', function () {
  const actions = document.getElementById('eventModalActions');
  if (actions) {
    actions.innerHTML = '';
    actions.setAttribute('aria-hidden', 'true');
  }
});

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

function showSetReminderModal(eventData) {
    const modalEl = document.getElementById("setReminderModal");
    const reminderModal = new bootstrap.Modal(modalEl);

    // 寫入事件 ID 與事件類型（強制成 'true'/'false' 字串）
    document.getElementById('reminderEventId').value = eventData.id;
    document.getElementById('reminderIsUserEvent').value = String(!!eventData.is_user_event);

    const dtInput = document.getElementById('reminderDateTime');

    // 設定 min（現在時刻，避免選過去）
    const now = new Date();
    // 調整為本地時區的 YYYY-MM-DDTHH:MM
    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
    dtInput.min = now.toISOString().slice(0, 16);

    // 預設提醒時間：事件當天 09:00（若你的 eventData 只有日期）
    const eventDate = eventData.date; // 例如 '2025-08-20'
    const defaultTime = '09:00';
    const initial = `${eventDate}T${defaultTime}`;
    dtInput.value = initial;

    reminderModal.show();
}

document.getElementById("set-reminder-form").addEventListener("submit", function(e) {
    e.preventDefault();
    const eventId = document.getElementById('reminderEventId').value.trim();
    const isUserEvent = document.getElementById('reminderIsUserEvent').value === 'true';
    const reminderTime = document.getElementById('reminderDateTime').value; // 'YYYY-MM-DDTHH:MM'

    fetch('/set_reminder/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: JSON.stringify({
            event_id: eventId,
            is_user_event: isUserEvent,
            reminder_time: reminderTime
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            alert("提醒設定成功！指定時間會寄送到您的 Gmail。");
            bootstrap.Modal.getInstance(document.getElementById("setReminderModal")).hide();
        } else {
            alert("提醒設定失敗：" + data.message);
        }
    })
    .catch(err => {
        console.error('設定提醒失敗:', err);
        alert('設定提醒時發生錯誤。');
    });
});

document.getElementById("test-gmail-btn").addEventListener("click", function() {
    fetch("/test-gmail-send/", {
        method: "GET",
        headers: {
            "X-CSRFToken": getCookie("csrftoken")
        }
    })
    .then(res => res.json())
    .then(data => {
        alert(data.success ? data.message : "發送失敗: " + data.message);
    })
    .catch(err => {
        console.error(err);
        alert("發送測試時發生錯誤");
    });
});

// 取得 CSRF Token
function getCookie(name) {
    const cookieStr = document.cookie;
    if (!cookieStr) return null;
    const cookies = cookieStr.split("; ").map(c => c.trim());
    const target = cookies.find(c => c.startsWith(name + "="));
    return target ? decodeURIComponent(target.split("=").slice(1).join("=")) : null;
}