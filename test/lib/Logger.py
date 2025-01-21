import logging


class LoggerHandler(logging.Logger):
    def __init__(self,
                 name="root",
                 level=logging.DEBUG,
                 file=None,
                 format="%(asctime)s - %(filename)s - %(lineno)d - %(name)s - %(levelname)s - %(message)s"):
        
        # 调用父类构造函数
        super().__init__(name)

        # 避免重复添加处理器
        if not self.handlers:
            # 设置日志级别
            self.setLevel(level)

            # 初始化日志格式
            fmt = logging.Formatter(format)

            # 如果提供了文件路径，初始化文件处理器
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

            # 初始化完成日志
            self.info("LoggerHandler initialized successfully.")


# 示例代码
if __name__ == "__main__":
    logger = LoggerHandler(name="my_logger", level=logging.INFO, file="example.log")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
