from asyncua import Client, ua
from lib import Logger
import asyncio
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

url = "opc.tcp://192.168.1.120:4840"
root_uri = "http://www.siemens.com/simatic-s7-opcua"

# 接受信号类
class Receive_data():
    def __init__(self, client, root_idx)-> None:
        self.task_id_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:send", f"3:task_id"]
        self.heart_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:send", f"3:heart"]
        self.action_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:send", f"3:action"]
        # self.request_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:send", f"3:request"]
        self.client = client

    async def init_node(self):
        self.node_task_id = await self.client.nodes.objects.get_child(self.task_id_node_str)
        self.node_heart = await self.client.nodes.objects.get_child(self.heart_node_str)
        self.node_action = await self.client.nodes.objects.get_child(self.action_node_str)
        # self.node_request = await self.client.nodes.objects.get_child(self.request_node_str)

    async def get_action(self):
        val = await self.node_action.get_value()
        return val

    async def get_heart(self):
        val = await self.node_heart.get_value()
        return val

    async def get_task_id(self):
        val = await self.node_task_id.get_value()
        return val


# 拆垛类
class Send_data():
    def __init__(self, client, root_idx):
        self.heart_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:heart"]
        self.task_id_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:task_id"]
        self.success_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:sucess"]
        # self.error_code_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:error_code"]
        self.X_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:X"]
        self.Y_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:Y"]
        self.Z_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:Z"]
        self.S_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:S"]

        self.num_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:num"]
        self.put_direction_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:put_direction"]
        self.response_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:response"]
        self.status_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:status"]
        self.pack_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:pack"]
        self.client = client

    async def init_node(self):
        self.node_heart = await self.client.nodes.objects.get_child(self.heart_node_str)
        self.node_task_id = await self.client.nodes.objects.get_child(self.task_id_node_str)
        self.node_success = await self.client.nodes.objects.get_child(self.success_node_str)
        # self.node_error_code = await self.client.nodes.objects.get_child(self.error_code_node_str)
        self.node_X = await self.client.nodes.objects.get_child(self.X_node_str)
        self.node_Y = await self.client.nodes.objects.get_child(self.Y_node_str)
        self.node_Z = await self.client.nodes.objects.get_child(self.Z_node_str)
        self.node_S = await self.client.nodes.objects.get_child(self.S_node_str)

        self.node_num = await self.client.nodes.objects.get_child(self.num_node_str)
        self.node_put_direction = await self.client.nodes.objects.get_child(self.put_direction_node_str)
        self.node_response = await self.client.nodes.objects.get_child(self.response_node_str)
        self.node_status = await self.client.nodes.objects.get_child(self.status_node_str)
        self.node_pack = await self.client.nodes.objects.get_child(self.pack_node_str)


    async def set_success(self, success_val):
        success_val = ua.DataValue(ua.Variant(success_val, ua.VariantType.Boolean))
        await self.node_success.set_value(success_val)

    async def set_response(self, response_val):
        response_val = ua.DataValue(ua.Variant(response_val, ua.VariantType.Boolean))
        await self.node_response.set_value(response_val)

    async def set_type_code(self, status_val, pack_val):
        status_val = ua.DataValue(ua.Variant(status_val, ua.VariantType.Byte))
        pack_val = ua.DataValue(ua.Variant(pack_val, ua.VariantType.Byte))
        await self.node_status.set_value(status_val)
        await self.node_pack.set_value(pack_val)

    async def set_position(self, task_id_val, box_num_val, x_val, y_val, z_val, s_val, box_direction_val):
        task_id_val = ua.DataValue(ua.Variant(task_id_val, ua.VariantType.Int16))
        box_num_val = ua.DataValue(ua.Variant(box_num_val, ua.VariantType.Int32))
        x_val = ua.DataValue(ua.Variant(x_val, ua.VariantType.Float))
        y_val = ua.DataValue(ua.Variant(y_val, ua.VariantType.Float))
        z_val = ua.DataValue(ua.Variant(z_val, ua.VariantType.Float))
        s_val = ua.DataValue(ua.Variant(s_val, ua.VariantType.Float))
        box_direction_val = ua.DataValue(ua.Variant(box_direction_val, ua.VariantType.Boolean))

        await self.node_task_id.set_value(task_id_val)
        # await self.node_num.set_value(box_num_val)
        await self.node_X.set_value(x_val)
        await self.node_Y.set_value(y_val)
        await self.node_Z.set_value(z_val)
        await self.node_S.set_value(s_val)
        # await self.node_put_direction.set_value(box_direction_val)


#心跳控制程序
async def heart_set_1s(handle):
    while True:
        try:
            await asyncio.sleep(1)
            await handle.set_heart(0)
            await asyncio.sleep(1)
            await handle.set_heart(1)
        except Exception as e:
            await asyncio.sleep(3)
            logger.info("无法写入心跳数据, 主设备掉线")
        # await send_handle.set_action(1)
        # await asyncio.sleep(0.02)


async def main():
    modbus_url = ("192.168.1.70", 502)
    url = "opc.tcp://192.168.1.120:4840"
    root_uri = "http://www.siemens.com/simatic-s7-opcua"
    client = Client(url=url)

    async with client:
        root_idx = await client.get_namespace_index(uri=root_uri)
        
        RECEIVE= Receive_data(client, root_idx)
        SEND= Send_data(client, root_idx)

        # 等待初始化节点完成
        await RECEIVE.init_node()
        await SEND.init_node()
        
        print("set_success\n")


        await SEND.set_position(1,2,3.0,5.0,7.0,4.0,True)

        # try:
        #     # 持续运行以监听数据更改
        #     while True:
        #         await asyncio.sleep(1)
        # finally:
        #     # 取消订阅并关闭会话
        #     pass

            
# 主函数
if __name__ == "__main__":
    # # 使用当前时间生成日志文件名
    # nowStr = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    # logfile_name = f"logs/log_{nowStr}.txt"
    # # 实例化LoggerHandler
    # logger = Logger.LoggerHandler(file=logfile_name)
    asyncio.run(main())

