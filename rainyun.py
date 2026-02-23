import logging
import os
import random
import re
import time
import subprocess
import sys
import json
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

import cv2
import ddddocr
import requests
import ICR

_http_session = None
_http_session_lock = threading.Lock()

def get_http_session():
    global _http_session
    if _http_session is None:
        with _http_session_lock:
            if _http_session is None:
                _http_session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=20,
                    max_retries=3
                )
                _http_session.mount('http://', adapter)
                _http_session.mount('https://', adapter)
    return _http_session

from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from webdriver_manager.chrome import ChromeDriverManager
    try:
        from webdriver_manager.core.utils import ChromeType
    except ImportError:
        try:
            from webdriver_manager.chrome import ChromeType
        except ImportError:
            ChromeType = None
except ImportError:
    print("webdriver_manager未安装，将使用备用方式")
    ChromeDriverManager = None
    ChromeType = None

try:
    from notify import send
    print("已加载通知模块 (notify.py)")
except ImportError:
    print("警告: 未找到 notify.py，将无法发送通知。")
    def send(*args, **kwargs):
        pass


def init_selenium(debug=False, headless=False):
    ops = webdriver.ChromeOptions()
    if headless or os.environ.get("GITHUB_ACTIONS", "false") == "true":
        for option in ['--headless', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']:
            ops.add_argument(option)
    ops.add_argument('--window-size=1920,1080')
    ops.add_argument('--disable-blink-features=AutomationControlled')
    ops.add_argument('--no-proxy-server')
    ops.add_argument('--lang=zh-CN')
    
    is_github_actions = os.environ.get("GITHUB_ACTIONS", "false") == "true"
    if debug and not is_github_actions:
        ops.add_experimental_option("detach", True)
    
    try:
        if ChromeDriverManager:
            if ChromeType and hasattr(ChromeType, 'GOOGLE'):
                manager = ChromeDriverManager(chrome_type=ChromeType.GOOGLE)
            else:
                manager = ChromeDriverManager()
            driver_path = manager.install()
            service = Service(driver_path)
            driver = webdriver.Chrome(service=service, options=ops)
            return driver
    except Exception as e:
        print(f"webdriver-manager失败: {e}")

    try:
        driver = webdriver.Chrome(options=ops)
        return driver
    except Exception:
        pass
        
    raise Exception("无法初始化Selenium WebDriver")

def generate_random_fingerprint():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    ]
    
    resolutions = [
        "1920,1080",
        "1920,1200",
        "2560,1440",
        "1440,900",
        "1366,768",
        "1536,864"
    ]
    
    languages = [
        "zh-CN",
        "zh-CN,zh;q=0.9,en;q=0.8",
        "en-US,en;q=0.9,zh;q=0.8"
    ]
    
    timezones = [
        "Asia/Shanghai",
        "Asia/Hong_Kong",
        "Asia/Taipei",
        "Asia/Singapore"
    ]
    
    return {
        "user_agent": random.choice(user_agents),
        "resolution": random.choice(resolutions),
        "language": random.choice(languages),
        "timezone": random.choice(timezones)
    }

def init_selenium(debug=False, headless=False, fingerprint=None):
    if fingerprint is None:
        fingerprint = generate_random_fingerprint()
    
    ops = webdriver.ChromeOptions()
    if headless or os.environ.get("GITHUB_ACTIONS", "false") == "true":
        for option in ['--headless', '--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']:
            ops.add_argument(option)
    
    ops.add_argument(f'--window-size={fingerprint["resolution"]}')
    ops.add_argument('--disable-blink-features=AutomationControlled')
    ops.add_argument('--no-proxy-server')
    ops.add_argument(f'--lang={fingerprint["language"]}')
    ops.add_argument(f'--user-agent={fingerprint["user_agent"]}')
    ops.add_argument('--disable-infobars')
    ops.add_argument('--disable-extensions')
    ops.add_argument('--disable-notifications')
    ops.add_argument('--disable-popup-blocking')
    ops.add_argument('--disable-save-password-bubble')
    ops.add_argument('--disable-translate')
    ops.add_argument('--disable-default-apps')
    ops.add_argument('--disable-sync')
    ops.add_argument('--metrics-recording-only')
    ops.add_argument('--no-first-run')
    ops.add_argument('--safebrowsing-disable-auto-update')
    ops.add_argument('--disable-background-networking')
    ops.add_argument('--disable-background-timer-throttling')
    ops.add_argument('--disable-backgrounding-occluded-windows')
    ops.add_argument('--disable-renderer-backgrounding')
    ops.add_argument('--disable-component-update')
    ops.add_argument('--disable-domain-reliability')
    
    prefs = {
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False,
        "profile.default_content_setting_values.notifications": 2
    }
    ops.add_experimental_option("prefs", prefs)
    ops.add_experimental_option("excludeSwitches", ["enable-automation"])
    ops.add_experimental_option('useAutomationExtension', False)
    
    is_github_actions = os.environ.get("GITHUB_ACTIONS", "false") == "true"
    if debug and not is_github_actions:
        ops.add_experimental_option("detach", True)
    
    if is_github_actions:
        try:
            print("GitHub Actions环境：使用系统ChromeDriver")
            driver = webdriver.Chrome(options=ops)
            return driver
        except Exception as e:
            print(f"系统ChromeDriver失败: {e}")
            raise Exception(f"GitHub Actions环境初始化Selenium失败: {e}")
    
    if ChromeDriverManager:
        try:
            print("尝试使用webdriver-manager...")
            if ChromeType and hasattr(ChromeType, 'GOOGLE'):
                manager = ChromeDriverManager(chrome_type=ChromeType.GOOGLE)
            else:
                manager = ChromeDriverManager()
            driver_path = manager.install()
            if os.path.isfile(driver_path) and not driver_path.endswith('.chromedriver'):
                import stat
                if not os.access(driver_path, os.X_OK):
                    os.chmod(driver_path, os.stat(driver_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                service = Service(driver_path)
                driver = webdriver.Chrome(service=service, options=ops)
                return driver
            else:
                print(f"webdriver-manager返回无效路径: {driver_path}")
        except Exception as e:
            print(f"webdriver-manager失败: {e}")

    try:
        print("尝试使用系统ChromeDriver...")
        driver = webdriver.Chrome(options=ops)
        return driver
    except Exception as e:
        print(f"系统ChromeDriver失败: {e}")
        
    raise Exception("无法初始化Selenium WebDriver")

def get_cookie_file_path(user):
    cookie_dir = "cookies"
    os.makedirs(cookie_dir, exist_ok=True)
    user_hash = hashlib.md5(user.encode()).hexdigest()
    return os.path.join(cookie_dir, f"{user_hash}.json")

def save_cookies(driver, user):
    try:
        cookie_file = get_cookie_file_path(user)
        cookies = driver.get_cookies()
        with open(cookie_file, 'w') as f:
            json.dump(cookies, f)
        logger.info(f"Cookie已保存: {cookie_file}")
        return True
    except Exception as e:
        logger.error(f"保存Cookie失败: {e}")
        return False

def load_cookies(driver, user):
    try:
        cookie_file = get_cookie_file_path(user)
        if not os.path.exists(cookie_file):
            logger.info("未找到Cookie缓存")
            return False
        
        with open(cookie_file, 'r') as f:
            cookies = json.load(f)
        
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except Exception:
                pass
        
        logger.info(f"Cookie已加载: {cookie_file}")
        return True
    except Exception as e:
        logger.error(f"加载Cookie失败: {e}")
        return False

def check_cookie_valid(driver):
    try:
        driver.get("https://app.rainyun.com/account/dashboard")
        time.sleep(2)
        return "dashboard" in driver.current_url or "login" not in driver.current_url
    except Exception:
        return False

def download_image(url, filename):
    os.makedirs("temp", exist_ok=True)
    try:
        session = get_http_session()
        response = session.get(url, timeout=10, proxies={"http": None, "https": None}, verify=False)
        if response.status_code == 200:
            with open(os.path.join("temp", filename), "wb") as f:
                f.write(response.content)
            return True
        return False
    except Exception as e:
        logger.error(f"下载图片异常: {str(e)}")
        return False


def get_url_from_style(style):
    return re.search(r'url\(["\']?(.*?)["\']?\)', style).group(1)


def get_width_from_style(style):
    return re.search(r'width:\s*([\d.]+)px', style).group(1)


def get_height_from_style(style):
    return re.search(r'height:\s*([\d.]+)px', style).group(1)


def process_captcha():
    global ocr, det, wait, driver
    
    try:
        download_captcha_img()
        if check_captcha():
            logger.info("开始使用ICR识别验证码")
            result = ICR.main("temp/captcha.jpg", "temp/sprite.jpg")
            if result and len(result) > 0:
                captcha = cv2.imread("temp/captcha.jpg")
                for info in result:
                    rect = info['bg_rect']
                    x, y = int(rect[0] + (rect[2] / 2)), int(rect[1] + (rect[3] / 2))
                    logger.info(f"图案 {info['sprite_idx'] + 1} 位于 ({x}, {y})")
                    slideBg = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="slideBg"]')))
                    style = slideBg.get_attribute("style")
                    width_raw, height_raw = captcha.shape[1], captcha.shape[0]
                    width, height = float(get_width_from_style(style)), float(get_height_from_style(style))
                    x_offset, y_offset = float(-width / 2), float(-height / 2)
                    final_x, final_y = int(x_offset + x / width_raw * width), int(y_offset + y / height_raw * height)
                    ActionChains(driver).move_to_element_with_offset(slideBg, final_x, final_y).click().perform()
                confirm = wait.until(
                    EC.element_to_be_clickable((By.XPATH, '//*[@id="tcStatus"]/div[2]/div[2]/div/div')))
                logger.info("提交验证码")
                confirm.click()
                time.sleep(5)
                result = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="tcOperation"]')))
                if result.get_attribute("class") == 'tc-opera pointer show-success':
                    logger.info("验证码通过")
                    return
                else:
                    logger.error("验证码未通过，正在重试")
            else:
                logger.error("ICR识别失败，正在重试")
        else:
            logger.error("当前验证码识别率低，尝试刷新")
        
        try:
             reload = driver.find_element(By.XPATH, '//*[@id="reload"]')
             time.sleep(2)
             reload.click()
             time.sleep(5)
             process_captcha()
        except:
             pass

    except TimeoutException:
        logger.error("获取验证码图片失败")
    except Exception as e:
        logger.error(f"处理验证码时发生错误: {e}")


def download_captcha_img():
    global wait
    
    if os.path.exists("temp"):
        for filename in os.listdir("temp"):
            file_path = os.path.join("temp", filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
    slideBg = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="slideBg"]')))
    img1_style = slideBg.get_attribute("style")
    img1_url = get_url_from_style(img1_style)
    logger.info("开始下载验证码图片(1): " + img1_url)
    download_image(img1_url, "captcha.jpg")
    sprite = wait.until(EC.visibility_of_element_located((By.XPATH, '//*[@id="instruction"]/div/img')))
    img2_url = sprite.get_attribute("src")
    logger.info("开始下载验证码图片(2): " + img2_url)
    download_image(img2_url, "sprite.jpg")


def check_captcha() -> bool:
    global ocr
    
    if ICR is None:
        try:
            raw = cv2.imread("temp/sprite.jpg")
            if raw is None: return False
            
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian < 50: return False
                
            h, w = raw.shape[:2]
            for i in range(3):
                w_segment = w // 3
                start_x = max(0, w_segment * i + 2)
                end_x = min(w, w_segment * (i + 1) - 2)
                temp = raw[:, start_x:end_x]
                cv2.imwrite(f"temp/sprite_{i + 1}.jpg", temp)
                
                with open(f"temp/sprite_{i + 1}.jpg", mode="rb") as f:
                    temp_rb = f.read()
                try:
                    result = ocr.classification(temp_rb)
                    if result in ["0", "1"]: return False
                except Exception:
                    return False
            return True
        except Exception:
            return False
    else:
        try:
            raw = cv2.imread("temp/sprite.jpg")
            if raw is None: return False
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
            return laplacian >= 30
        except Exception:
            return False


def check_answer(d: dict) -> bool:
    flipped = dict()
    for key in d.keys():
        flipped[d[key]] = key
    if len(d.values()) != len(flipped.keys()):
        return False
    return True


def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return morph

def compute_similarity(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None: return 0.0, 0
    
    scale = 100.0 / max(img1.shape) if max(img1.shape) > 100 else 1.0
    img1 = cv2.resize(img1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    scale = 100.0 / max(img2.shape) if max(img2.shape) > 100 else 1.0
    img2 = cv2.resize(img2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)

    try:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        if des1 is None or des2 is None: return 0.0, 0

        flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matches = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.7 * n.distance]
        
        if len(good) == 0: return 0.0, 0
        feature_factor = min(1.0, len(kp1) / 100.0, len(kp2) / 100.0)
        match_ratio = len(good) / min(len(des1), len(des2))
        return match_ratio * 0.7 + feature_factor * 0.3, len(good)
    except Exception:
        return 0.0, 0


def sign_in_account(user, pwd, debug=False, headless=False):
    timeout = 15
    driver = None
    
    global ocr, det, wait 
    
    try:
        logger.info(f"开始处理账户: {user}")
        
        if ICR is not None:
            logger.info("使用ICR模块进行验证码识别（旋转分析+模板匹配）")
        else:
            logger.info("初始化 ddddocr")
            ocr = ddddocr.DdddOcr(ocr=True, show_ad=False)
            det = ddddocr.DdddOcr(det=True, show_ad=False)
        
        fingerprint = generate_random_fingerprint()
        logger.info(f"使用浏览器指纹: {fingerprint['user_agent'][:50]}...")
        logger.info("初始化 Selenium")
        driver = init_selenium(debug=debug, headless=headless, fingerprint=fingerprint)
        
        globals()['driver'] = driver 
        wait = WebDriverWait(driver, timeout) 
        
        try:
            with open("stealth.min.js", mode="r") as f: js = f.read()
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": js})
        except: pass
        
        # 尝试使用Cookie免密登录
        logger.info("尝试使用Cookie缓存登录")
        driver.get("https://app.rainyun.com")
        if load_cookies(driver, user):
            driver.refresh()
            time.sleep(3)
            if check_cookie_valid(driver):
                logger.info("Cookie登录成功！")
                logger.info("正在转到赚取积分页")
                try:
                    driver.get("https://app.rainyun.com/account/reward/earn")
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
                    time.sleep(3)
                    
                    # === 核心修复 1 (Cookie分支)：精准视觉定位 ===
                    try:
                        btn_xpath = "//span[contains(text(),'每日签到')]/following::*[contains(text(),'领取奖励')][1]"
                        claim_btns = driver.find_elements(By.XPATH, btn_xpath)
                        
                        if claim_btns and claim_btns[0].is_displayed():
                            logger.info("🎯 检测到【领取奖励】绿色徽章，准备签到！")
                            earn = claim_btns[0]
                            
                            driver.execute_script("arguments[0].scrollIntoView(true);", earn)
                            time.sleep(1)
                            logger.info("点击领取奖励")
                            driver.execute_script("arguments[0].click();", earn)
                            
                            logger.info("等待验证码加载（如果有）...")
                            try:
                                WebDriverWait(driver, 15, poll_frequency=0.25).until(
                                    EC.visibility_of_element_located((By.ID, "tcaptcha_iframe_dy"))
                                )
                                wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "tcaptcha_iframe_dy")))
                                logger.info("处理验证码")
                                process_captcha()
                                driver.switch_to.default_content()
                            except TimeoutException:
                                logger.info("未触发验证码，签到成功！")
                                driver.switch_to.default_content()
                            except Exception as e:
                                logger.error(f"验证码处理过程出错: {e}")
                                driver.switch_to.default_content()
                                
                            logger.info("领取奖励操作完成")
                            
                        else:
                            logger.info("✅ 未找到【领取奖励】按钮，‘每日签到’已完成，跳过。")
                            try:
                                points_raw = driver.find_element(By.XPATH, '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[1]/div[1]/div/p/div/h3').get_attribute("textContent")
                                current_points = int(''.join(re.findall(r'\d+', points_raw)))
                            except:
                                current_points = 0
                            return True, user, current_points, None
                            
                    except Exception as e:
                        logger.error(f"签到状态判断或点击出错: {e}")
                        
                except Exception as e:
                    logger.error(f"Cookie登录后操作出错: {e}")
                    pass
            else:
                logger.info("Cookie已失效，尝试重新登录")
        
        logger.info("发起登录请求")
        driver.get("https://app.rainyun.com/auth/login")
        wait = WebDriverWait(driver, timeout)
        
        # 登录流程
        username = wait.until(EC.visibility_of_element_located((By.NAME, 'login-field')))
        password = wait.until(EC.visibility_of_element_located((By.NAME, 'login-password')))
        try:
            login_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="app"]/div[1]/div[1]/div/div[2]/fade/div/div/span/form/button')))
        except:
            login_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[type="submit"]')))
            
        username.clear()
        password.clear()
        username.send_keys(user)
        time.sleep(0.5)
        password.send_keys(pwd)
        time.sleep(0.5)
        driver.execute_script("arguments[0].click();", login_button)
        
        # 登录验证码
        try:
            wait.until(EC.visibility_of_element_located((By.ID, 'tcaptcha_iframe_dy')))
            logger.warning("触发验证码！")
            driver.switch_to.frame("tcaptcha_iframe_dy")
            process_captcha()
        except TimeoutException:
            logger.info("未触发验证码")
        
        time.sleep(5)
        driver.switch_to.default_content()
        
        if "dashboard" in driver.current_url:
            logger.info("登录成功！")
            save_cookies(driver, user)
            logger.info("正在转到赚取积分页")
            
            for _ in range(3):
                try:
                    driver.get("https://app.rainyun.com/account/reward/earn")
                    wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
                    time.sleep(3)

                    # === 核心修复 2 (密码分支)：精准视觉定位 ===
                    try:
                        btn_xpath = "//span[contains(text(),'每日签到')]/following::*[contains(text(),'领取奖励')][1]"
                        claim_btns = driver.find_elements(By.XPATH, btn_xpath)
                        
                        if claim_btns and claim_btns[0].is_displayed():
                            logger.info("🎯 检测到【领取奖励】绿色徽章，准备签到！")
                            earn = claim_btns[0]
                            
                            driver.execute_script("arguments[0].scrollIntoView(true);", earn)
                            time.sleep(1)
                            logger.info("点击领取奖励")
                            driver.execute_script("arguments[0].click();", earn)
                            
                            logger.info("等待验证码加载（如果有）...")
                            try:
                                WebDriverWait(driver, 15, poll_frequency=0.25).until(
                                    EC.visibility_of_element_located((By.ID, "tcaptcha_iframe_dy"))
                                )
                                wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, "tcaptcha_iframe_dy")))
                                logger.info("处理验证码")
                                process_captcha()
                                driver.switch_to.default_content()
                            except TimeoutException:
                                logger.info("未触发验证码，签到成功！")
                                driver.switch_to.default_content()
                            except Exception as e:
                                logger.error(f"验证码处理过程出错: {e}")
                                driver.switch_to.default_content()
                                
                            logger.info("领取奖励操作完成")
                            break
                            
                        else:
                            logger.info("✅ 未找到【领取奖励】按钮，‘每日签到’已完成，跳过。")
                            try:
                                points_raw = driver.find_element(By.XPATH, '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[1]/div[1]/div/p/div/h3').get_attribute("textContent")
                                current_points = int(''.join(re.findall(r'\d+', points_raw)))
                            except:
                                current_points = 0
                            return True, user, current_points, None
                            
                    except Exception as e:
                        logger.error(f"签到状态判断或点击出错: {e}")
                        driver.refresh()
                        time.sleep(3)
                except Exception as e:
                    logger.error(f"出错: {e}")
                    time.sleep(3)
        else:
            return False, user, 0, "登录失败"

        driver.implicitly_wait(5)
        # 简单的积分获取
        try:
            points_raw = driver.find_element(By.XPATH, '//*[@id="app"]/div[1]/div[3]/div[2]/div/div/div[2]/div[1]/div[1]/div/p/div/h3').get_attribute("textContent")
            current_points = int(''.join(re.findall(r'\d+', points_raw)))
            logger.info(f"当前剩余积分: {current_points} | 约为 {current_points / 2000:.2f} 元")
        except:
            current_points = 0
            
        logger.info("任务执行成功！")
        return True, user, current_points, None

    except Exception as e:
        logger.error(f"异常: {str(e)}", exc_info=True)
        return False, user, 0, str(e)
    finally:
        if driver:
            try: driver.quit()
            except: pass

if __name__ == "__main__":
    is_github_actions = os.environ.get("GITHUB_ACTIONS", "false") == "true"
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    headless = os.environ.get('HEADLESS', 'false').lower() == 'true'
    if is_github_actions: headless = True
    
    max_workers = int(os.environ.get('MAX_WORKERS', '2'))
    max_retries = int(os.environ.get('MAX_RETRIES', '1'))
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    ocr = None
    det = None
    wait = None

    ver = "2.6 (Ultimate Visual Match based on Screenshots)"
    logger.info("------------------------------------------------------------------")
    logger.info(f"雨云自动签到工作流 v{ver}")
    logger.info(f"最大并发线程数: {max_workers}")
    logger.info(f"最大重试次数: {max_retries}")
    logger.info("------------------------------------------------------------------")
    
    accounts = []
    users_env = os.environ.get("RAINYUN_USER", "")
    passwords_env = os.environ.get("RAINYUN_PASS", "")
    users = [user.strip() for user in users_env.split('\n') if user.strip()]
    passwords = [pwd.strip() for pwd in passwords_env.split('\n') if pwd.strip()]
    
    if len(users) == len(passwords) and len(users) > 0:
        for user, pwd in zip(users, passwords):
            accounts.append((user, pwd))
    else:
        logger.error("未找到有效账户配置或数量不匹配")
        exit(1)
    
    results = []
    
    def process_account(account_info):
        index, user, pwd = account_info
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] === 开始处理第 {index} 个账户: {user} ===")
        result = sign_in_account(user, pwd, debug=debug, headless=headless)
        logger.info(f"[{thread_name}] === 第 {index} 个账户处理完成 ===")
        return (index, result)
    
    current_retry = 0
    failed_accounts = [(i + 1, user, pwd) for i, (user, pwd) in enumerate(accounts)]
    
    while current_retry <= max_retries and failed_accounts:
        if current_retry > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"第 {current_retry} 轮重试，共 {len(failed_accounts)} 个失败账户")
            logger.info(f"{'='*60}\n")
            time.sleep(random.randint(5, 15))
        else:
            logger.info(f"开始并发处理 {len(failed_accounts)} 个账户...")
        
        account_infos = failed_accounts.copy()
        failed_accounts = []
        
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="Worker") as executor:
            future_to_account = {executor.submit(process_account, info): info for info in account_infos}
            
            for future in as_completed(future_to_account):
                account_info = future_to_account[future]
                try:
                    index, result = future.result()
                    results.append((index, result))
                    if not result[0]:
                        failed_accounts.append(account_info)
                except Exception as e:
                    logger.error(f"账户 {account_info[1]} 处理异常: {e}")
                    results.append((account_info[0], (False, account_info[1], 0, str(e))))
                    failed_accounts.append(account_info)
        
        current_retry += 1
    
    results.sort(key=lambda x: x[0])
    results = [result for _, result in results]
    
    logger.info("\n所有账户处理完成，生成统一通知...")
    
    success_count = sum(1 for r in results if r[0])
    total_count = len(results)
    
    if success_count == total_count:
        notification_title = f"✅ 雨云自动签到完成 - 全部成功"
    elif success_count > 0:
        notification_title = f"⚠️ 雨云自动签到完成 - 部分成功 ({success_count}/{total_count})"
    else:
        notification_title = f"❌ 雨云自动签到完成 - 全部失败"
    
    notification_content = f"雨云自动签到结果汇总：\n\n总账户数: {total_count}\n成功账户数: {success_count}\n失败账户数: {total_count - success_count}\n\n详细结果：\n"
    
    for i, (success, user, points, error_msg) in enumerate(results, 1):
        if success:
            notification_content += f"{i}. ✅ {user}\n   积分: {points} | 约 {points / 2000:.2f} 元\n"
        else:
            notification_content += f"{i}. ❌ {user}\n   错误: {error_msg}\n"
    
    try:
        send(notification_title, notification_content)
        logger.info("统一通知发送成功")
    except Exception as e:
        logger.error(f"发送通知失败: {e}")
