from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import random

# Cài đặt số lượng form bạn muốn điền
SO_LAN_DIEN = 22

# 1. Khởi tạo trình duyệt MỘT LẦN ở ngoài vòng lặp
driver = webdriver.Chrome()
wait = WebDriverWait(driver, 10)
url = "https://docs.google.com/forms/d/e/1FAIpQLScb-8QxgvCWgEyA6zXrp0cf5sB27AznIotnQIYZTbGZ4Gkaxg/viewform?fbzx=7410593511366298601"

try:
    for i in range(SO_LAN_DIEN):
        print(f"\n--- Đang thực hiện lần thứ {i + 1}/{SO_LAN_DIEN} ---")
        
        # Mở trang form (hoặc tải lại form mới)
        driver.get(url)

        # ==========================================
        # XỬ LÝ TRANG 1
        # ==========================================
        # Tìm nút "Tiếp" (hoặc "Next")
        next_btn_xpath = "//*[contains(text(), 'Tiếp') or contains(text(), 'Next')]"
        next_button = wait.until(EC.element_to_be_clickable((By.XPATH, next_btn_xpath)))
        next_button.click()
        print("Đã bấm nút Tiếp.")

        # ==========================================
        # XỬ LÝ TRANG 2
        # ==========================================
        # Tạo index ngẫu nhiên và chọn đáp án
        x = random.randint(1, 3)
        if i >= 5 and i <=15: 
            x = 3
        if i >= 18 and i <= 22:
            x = 3
        # if i > 22:
        #     x = 2
        option_xpath = f"//*[@id='i{3*(x+1)}']/div[3]/div" 
        
        option = wait.until(EC.element_to_be_clickable((By.XPATH, option_xpath)))
        option.click()
        print(f"Đã chọn đáp án ngẫu nhiên (id=i{3*(x+1)}).")

        # Tìm và bấm nút "Gửi" (Submit)
        submit_btn_xpath = "//*[@id='mG61Hd']/div[2]/div/div[3]/div/div[1]/div[2]/span"
        submit_button = wait.until(EC.element_to_be_clickable((By.XPATH, submit_btn_xpath)))
        submit_button.click()
        
        print(f"Đã submit form thành công lần {i + 1}!")
        
        # Tạm dừng khoảng 1-2 giây để Google kịp ghi nhận response trước khi vòng lặp quay lại tải trang mới
        time.sleep(1.5)

except Exception as e:
    print(f"Lỗi ở vòng lặp thứ {i + 1}: {e}")

finally:
    # 2. Chỉ đóng trình duyệt khi đã chạy xong toàn bộ vòng lặp
    print("\nĐã hoàn thành toàn bộ quá trình. Đóng trình duyệt.")
    driver.quit()