import logging

class LoggerHandler(logging.Logger):
    def __init__(self,
                 name="root",
                 level=logging.DEBUG,  # 使用logging.DEBUG而非"DEBUG"
                 file=None,
                 format="%(asctime)s - %(filename)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s"):
        
        super().__init__(name)

        # 设置日志级别
        self.setLevel(level)

        # 初始化格式
        fmt = logging.Formatter(format)

        # 如果提供了文件路径，则初始化文件处理器
        if file:
            file_handler = logging.FileHandler(file)
            file_handler.setLevel(level)
            file_handler.setFormatter(fmt)
            self.addHandler(file_handler)

        # 初始化控制台处理器
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(fmt)
        self.addHandler(stream_handler)
        self.info("LoggerHandler init success.")




