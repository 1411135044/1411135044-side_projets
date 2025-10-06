// 初始化音效
const sounds = {
    correct: new Howl({ src: ['/static/sounds/correct.mp3'], volume: 0.7 }),
    wrong: new Howl({ src: ['/static/sounds/wrong.mp3'], volume: 0.7 })
};

// Socket.IO 連接
const socket = io();

// 遊戲狀態控制
let gameState = {
    score: 0,
    timeLeft: 60, // 遊戲總時長 (秒)
    isPlaying: false,
    timerId: null,
    currentGarbage: null // 當前垃圾物件
};

// 垃圾分類定義
const garbageTypes = [
    {
        name: '一般垃圾',
        images: [ // 注意這裡從 'image' 改為 'images' 並且是一個陣列
            '/static/images/general_waste_1.png',
            '/static/images/general_waste_2.png',
            '/static/images/general_waste_3.png'
        ],
        gesture: '1'
    },
    {
        name: '紙容器',
        images: [ // 注意這裡從 'image' 改為 'images' 並且是一個陣列
            '/static/images/paper_container_1.png',
            '/static/images/paper_container_2.png',
            '/static/images/paper_container_3.jpg',
            '/static/images/paper_container_4.jpg'
        ],
        gesture: '2'
    },
    {
        name: '塑膠類',
        images: [ // 注意這裡從 'image' 改為 'images' 並且是一個陣列
            '/static/images/plastic_1.png',
            '/static/images/plastic_2.png',
            '/static/images/plastic_3.png'
        ],
        gesture: '3'
    },
    {
        name: '寶特瓶',
        images: [ // 注意這裡從 'image' 改為 'images' 並且是一個陣列
            '/static/images/pet_bottle_1.jpg',
            '/static/images/pet_bottle_2.jpg',
            '/static/images/pet_bottle_3.jpg'
        ],
        gesture: '4'
    },
    {
        name: '鐵鋁罐',
        images: [ // 注意這裡從 'image' 改為 'images' 並且是一個陣列
            '/static/images/cans_1.png',
            '/static/images/cans_2.jpg',
            '/static/images/cans_3.jpeg'
        ],
        gesture: '5'
    },
    {
        name: '紙類',
        images: [ // 注意這裡從 'image' 改為 'images' 並且是一個陣列
            '/static/images/paper_1.png',
            '/static/images/paper_2.jpg',
            '/static/images/paper_3.png'
        ],
        gesture: '6'
    },
    {
        name: '金屬類',
        images: [ // 注意這裡從 'image' 改為 'images' 並且是一個陣列
            '/static/images/metal_1.jpg',
            '/static/images/metal_2.jpeg',
            '/static/images/metal_3.jpg',
            '/static/images/metal_4.jpg'
        ],
        gesture: '7'
    },
    {
        name: '玻璃類',
        images: [ // 注意這裡從 'image' 改為 'images' 並且是一個陣列
            '/static/images/glass_1.png',
            '/static/images/glass_2.png',
            '/static/images/glass_3.png',
            '/static/images/glass_4.png'
        ],
        gesture: '8'
    }
];
let availableGarbageTypes = []; // 用於隨機選取時不重複，直到所有類型都出現過

// 元素快取
const elements = {
    startScreen: document.getElementById('start-screen'),
    gameScreen: document.getElementById('game-screen'),
    resultScreen: document.getElementById('result-screen'),
    garbageImage: document.getElementById('garbage-image'), // 新增
    playerGesture: document.getElementById('player-gesture'),
    gestureCountdownText: document.getElementById('gesture-countdown-text'), // 新增用於顯示手勢倒數
    timerFill: document.querySelector('.timer-fill'),
    currentScore: document.getElementById('current-score'),
    finalScore: document.getElementById('final-score'),
    btnStart: document.getElementById('btn-start'),
    btnRestart: document.getElementById('btn-restart'),
    btnHome: document.getElementById('btn-home'),
    gameTimeLeft: document.getElementById('game-time-left'), // 顯示遊戲剩餘時間
    categoryLegend: document.getElementById('category-legend-list') // 分類提示列表
};

// 事件綁定
elements.btnStart.addEventListener('click', startGame);
elements.btnRestart.addEventListener('click', restartGame);
elements.btnHome.addEventListener('click', goToHome);

// Socket 數據監聽
socket.on('gesture', data => {
    if (!gameState.isPlaying) return;

    updatePlayerGesture(data.gesture);
    updateGestureCountdown(data.remaining, data.confirmed);

    if (data.confirmed && data.gesture !== '') { // 確保確認的手勢不是空的
        validateGesture(data.gesture);
    }
});

// 遊戲核心邏輯
function startGame() {
    gameState = {
        score: 0,
        timeLeft: 60, // 重置遊戲時間
        isPlaying: true,
        timerId: null,
        currentGarbage: null
    };
    availableGarbageTypes = [...garbageTypes]; // 複製一份用於隨機選取

    elements.startScreen.classList.remove('active');
    elements.resultScreen.style.display = 'none';
    elements.gameScreen.classList.add('active');

    elements.currentScore.textContent = '0';
    elements.timerFill.style.width = '100%';
    elements.gameTimeLeft.textContent = gameState.timeLeft;
    elements.playerGesture.textContent = '?';
    elements.gestureCountdownText.textContent = '準備手勢';


    generateNewTarget();
    startTimer();
    populateCategoryLegend(); // 填充分類提示
}

function populateCategoryLegend() {
    elements.categoryLegend.innerHTML = ''; // 清空舊的
    garbageTypes.forEach(type => {
        const listItem = document.createElement('li');
        listItem.textContent = `手勢 ${type.gesture}: ${type.name}`;
        elements.categoryLegend.appendChild(listItem);
    });
}

function generateNewTarget() {
    if (availableGarbageTypes.length === 0) {
        availableGarbageTypes = [...garbageTypes]; // 或者其他處理方式
        // endGame(); // 如果所有類型都至少展示過一次其所有圖片，可以考慮結束
        // return;
    }

    const randomTypeIndex = Math.floor(Math.random() * availableGarbageTypes.length);
    // 注意：這裡不再從 availableGarbageTypes 中移除，除非您有特殊邏輯
    // 如果希望每個分類的所有圖片都至少出現一次才算完成一輪，邏輯會更複雜
    gameState.currentGarbage = availableGarbageTypes[randomTypeIndex];

    if (gameState.currentGarbage && gameState.currentGarbage.images && gameState.currentGarbage.images.length > 0) {
        // 從該分類的圖片陣列中隨機選擇一張圖片
        const randomImageIndex = Math.floor(Math.random() * gameState.currentGarbage.images.length);
        elements.garbageImage.src = gameState.currentGarbage.images[randomImageIndex];
        elements.garbageImage.alt = gameState.currentGarbage.name; // alt 仍然是分類名稱
        elements.garbageImage.style.display = 'block';
        // console.log(`新目標: ${gameState.currentGarbage.name}, 圖片: ${elements.garbageImage.src}, 手勢: ${gameState.currentGarbage.gesture}`);
    } else {
        console.error("無法生成新的垃圾目標或該分類沒有圖片！");
        // 可能需要一個備用圖片或處理邏輯
        // endGame();
        return; // 提前返回，避免後續錯誤
    }
    elements.gestureCountdownText.textContent = '辨識中...';
    elements.gestureCountdownText.style.color = '#fff';
}

function validateGesture(playerGesture) {
    if (!gameState.isPlaying || !gameState.currentGarbage) return;

    // console.log(`驗證手勢: 玩家 ${playerGesture}, 目標 ${gameState.currentGarbage.gesture}`);

    if (playerGesture === gameState.currentGarbage.gesture) {
        gameState.score += 10;
        elements.currentScore.textContent = gameState.score;

        elements.garbageImage.classList.add('correct-effect');
        setTimeout(() => {
            elements.garbageImage.classList.remove('correct-effect');
        }, 300); // 縮短特效時間
        sounds.correct.play();
        generateNewTarget();
    } else {
        // 答錯不扣分，但播放音效並產生新目標
        elements.garbageImage.classList.add('wrong-effect'); // 可以給圖片或玩家手勢加上晃動效果
         setTimeout(() => {
            elements.garbageImage.classList.remove('wrong-effect');
        }, 300);
        sounds.wrong.play();
        // 答錯後是否立即更換題目，或者給予一點懲罰時間，取決於設計
        // generateNewTarget(); // 這裡選擇答錯也更換題目
    }
}

function updatePlayerGesture(gesture) {
    elements.playerGesture.textContent = gesture || '?';
    elements.playerGesture.style.color = gesture ? '#2ecc71' : '#e74c3c'; // 綠色表示有偵測到，紅色表示無
}

function updateGestureCountdown(remaining, confirmed) {
    if (confirmed) {
        elements.gestureCountdownText.textContent = '確認!';
        elements.gestureCountdownText.style.color = '#2ecc71'; // 確認時綠色
    } else if (elements.playerGesture.textContent !== '?') { // 只有在偵測到手勢時才顯示倒數
        elements.gestureCountdownText.textContent = `維持 ${remaining.toFixed(1)}s`;
        elements.gestureCountdownText.style.color = '#e9warning'; // 倒數時黃色 (或其他顏色)
    } else {
        elements.gestureCountdownText.textContent = '辨識中...';
        elements.gestureCountdownText.style.color = '#fff'; // 預設白色
    }
}


function startTimer() {
    clearInterval(gameState.timerId);
    const initialTime = gameState.timeLeft; // 記住初始時間以計算百分比

    gameState.timerId = setInterval(() => {
        gameState.timeLeft -= 0.1;
        elements.gameTimeLeft.textContent = Math.max(0, Math.ceil(gameState.timeLeft)); // 更新遊戲剩餘時間顯示
        elements.timerFill.style.width = `${Math.max(0, (gameState.timeLeft / initialTime)) * 100}%`;

        if (gameState.timeLeft <= 0) {
            endGame();
        }
    }, 100);
}

function endGame() {
    if (!gameState.isPlaying) return; // 防止重複調用 endGame
    gameState.isPlaying = false;
    clearInterval(gameState.timerId);

    elements.gameScreen.classList.remove('active');
    elements.resultScreen.style.display = 'flex';
    elements.finalScore.textContent = gameState.score;
    elements.garbageImage.style.display = 'none'; // 隱藏垃圾圖片
    elements.playerGesture.textContent = '?';
    elements.gestureCountdownText.textContent = '遊戲結束';
}

function goToHome() {
    gameState.isPlaying = false;
    clearInterval(gameState.timerId);
    elements.resultScreen.style.display = 'none';
    elements.gameScreen.classList.remove('active');
    elements.startScreen.classList.add('active');
    elements.garbageImage.style.display = 'none';
    elements.playerGesture.textContent = '?';
    elements.gestureCountdownText.textContent = '';
    elements.currentScore.textContent = '0';

}

function restartGame() {
    elements.resultScreen.style.display = 'none';
    startGame();
}

// 初始化時隱藏遊戲和結果畫面
document.addEventListener('DOMContentLoaded', () => {
    elements.gameScreen.classList.remove('active');
    elements.resultScreen.style.display = 'none';
    elements.startScreen.classList.add('active');
    elements.garbageImage.style.display = 'none'; // 初始隱藏圖片
    elements.gestureCountdownText.textContent = ''; // 初始清空倒數文本
});

// 動態添加 CSS 特效 (如果 style.css 中沒有的話)
const dynamicStyle = document.createElement('style');
dynamicStyle.textContent = `
    #garbage-image.correct-effect {
        animation: correctBlink 0.3s ease-in-out;
        border: 5px solid #2ecc71; /* 答對時加綠色邊框 */
    }
    #garbage-image.wrong-effect {
        animation: wrongShake 0.3s ease-in-out;
        border: 5px solid #e74c3c; /* 答錯時加紅色邊框 */
    }
    @keyframes correctBlink {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    @keyframes wrongShake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-8px) rotate(-2deg); }
        75% { transform: translateX(8px) rotate(2deg); }
    }
`;
document.head.appendChild(dynamicStyle);