from opcua import Server
import random
import time
# 创建并配置服务器
server = Server()
server.set_endpoint("opc.tcp://0.0.0.0:4840")
idx = server.register_namespace("http://www.siemens.com/simatic-s7-opcua")
# 添加对象和变量
objects_node = server.get_objects_node()
myobj = objects_node.add_object(idx, "Sinumerik")
var1 = myobj.add_variable(idx, "RandomValue1", 0)
var2 = myobj.add_variable(idx, "RandomValue2", 0)
obj = myobj.add_object(idx, "Test")
var3 = obj.add_variable(idx, "RandomValue3", 0)
# 启动服务器
server.start()
print("Server started at opc.tcp://0.0.0.0:4840")
# 无限循环，更新变量值
try:
    while True:
        var1.set_value(random.randint(0, 100))
        var2.set_value(random.randint(0, 100))
        var3.set_value(random.randint(0, 100))
        print("Variables updated")
        time.sleep(5)
except KeyboardInterrupt:
    server.stop()
