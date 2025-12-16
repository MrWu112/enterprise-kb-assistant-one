import os
import base64 # 用于对二进制数据（如 salt、密钥）进行 Base64 编码/解码，便于存储为字符串
import hashlib # 提供哈希函数（如 SHA256）和 PBKDF2 密钥派生函数。
import hmac # 提供安全的比较函数 hmac.compare_digest()，防止时序攻击（timing attack）。
import secrets # 用于生成密码学安全的随机数（如 salt），比 random 更安全。
from datetime import datetime, timedelta, timezone # 用于处理 JWT 的过期时间（exp 字段）。
from typing import Any

from jose import jwt # 使用 python-jose 库来生成和解析 JWT（JSON Web Token）。


# ===== Password hashing (no passlib/bcrypt) =====

# 格式：pbkdf2_sha256$<iters>$<salt_b64>$<dk_b64>
# 定义哈希字符串的格式，类似 Django 的密码格式：
# pbkdf2_sha256：算法名
# <iters>：迭代次数
# <salt_b64>：Base64 编码的 salt
# <dk_b64>：Base64 编码的派生密钥（derived key）
PWD_SCHEME = "pbkdf2_sha256" # 指定使用的哈希方案名称。
PWD_ITERS = int(os.getenv("PWD_ITERS", "310000"))  # 可按机器性能调整 # 从环境变量读取 PBKDF2 迭代次数，默认 310000（符合 OWASP 推荐值）。
                                                   # 迭代次数越高越安全，但计算更慢。
PWD_SALT_BYTES = 16 # Salt 长度为 16 字节（128 位），足够随机且高效。
PWD_DKLEN = 32  # 256-bit # 派生密钥长度为 32 字节（256 位），匹配 SHA256 输出。

def _b64e(b: bytes) -> str:
    # _b64e：将 bytes 转为 URL 安全的 Base64 字符串，并移除尾部填充 =（节省空间，常见于 token）。
    # urlsafe_b64encode 使用 - 和 _ 而非 + 和 /，避免 URL 编码问题。
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")

def _b64d(s: str) -> bytes:
    # _b64d：将无填充的 Base64 字符串还原为 bytes。
    # 自动补全缺失的 = 填充（因为 Base64 长度必须是 4 的倍数）。
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))

def hash_password(password: str) -> str:
    """
    密码哈希与验证
    输入明文密码，返回哈希字符串。
    """
    salt = secrets.token_bytes(PWD_SALT_BYTES) # 生成 16 字节的加密安全随机 salt。
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PWD_ITERS, dklen=PWD_DKLEN)
    # 使用 PBKDF2-HMAC-SHA256 算法：
    # password.encode("utf-8")：将密码转为字节
    # salt：随机盐
    # PWD_ITERS：迭代次数（防暴力破解）
    # dklen=32：输出 32 字节密钥
    return f"{PWD_SCHEME}${PWD_ITERS}${_b64e(salt)}${_b64e(dk)}" # 按格式拼接成字符串

def verify_password(plain: str, hashed: str) -> bool:
    """
    验证明文密码是否匹配哈希值。
    :param plain:
    :param hashed:
    :return:
    """
    try:
        scheme, iters_s, salt_b64, dk_b64 = hashed.split("$", 3) # 将哈希字符串按 $ 分割为 4 部分（最多分割 3 次）。
        if scheme != PWD_SCHEME:
            # 如果算法不匹配，直接拒绝（未来可扩展其他算法）
            return False
        iters = int(iters_s) # 解析迭代次数
        salt = _b64d(salt_b64) # 解析salt
        dk_expected = _b64d(dk_b64) # 解析预期密钥
        dk = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, iters, dklen=len(dk_expected)) # 用相同参数重新计算派生密钥
        return hmac.compare_digest(dk, dk_expected) # 安全比较两个字节串，防止时序攻击（即使一个字节不同，耗时也相同）
    except Exception:
        # 任何解析错误（如格式不对）都返回 False，避免信息泄露
        return False

# ===== JWT =====
JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-change-me") # 从环境变量读取 JWT 签名密钥
JWT_ALG = os.getenv("JWT_ALG", "HS512")  # 使用 HMAC-SHA512 算法（比 HS256 更安全，输出 512 位）
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "120")) # Token 默认 120 分钟过期

def create_access_token(payload: dict[str, Any], expires_minutes: int | None = None) -> str:
    """
    创建 JWT 访问令牌
    """
    minutes = expires_minutes or JWT_EXPIRE_MINUTES # 使用传入的过期时间，或默认值。
    exp = datetime.now(timezone.utc) + timedelta(minutes=minutes) # 计算过期时间（UTC 时区，避免时区混乱）。
    to_encode = {**payload, "exp": exp} # 合并用户 payload 和标准 exp 声明。
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG) # 使用 python-jose 生成签名 JWT。

def decode_token(token: str) -> dict[str, Any]:
    """
    解析并验证 JWT。
    自动验证签名、过期时间等。
    如果无效（篡改、过期），会抛出异常（由调用者处理）。
    """
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])

if __name__ == "__main__":
    h1 = hash_password("123456")
    h2 = hash_password("123456")
    print(h1)
    print(h2)
    print(verify_password("123456", h1))

    d = {"username": "tom", "role": "admin"}
    t = create_access_token(d)
    print(t)
    print(decode_token(t))
