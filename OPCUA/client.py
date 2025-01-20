from asyncua import Client, ua
from lib import Logger
import asyncio
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

url = "opc.tcp://localhost:4840"
root_uri = "http://www.siemens.com/simatic-s7-opcua"

class action():
    def __init__(self,client,root_idx) -> None:
        self.heart_node_str = [f"3:PLC_m",f"3:DataBlockGlobal,",f"3:CCD",f"3:send",f"3:heart"]
        self.task_id_node_str = [f"3:PLC_m",f"3:DataBlockGlobal",f"3:CCD",f"3:send",f"3:task_id"]
        self.action_node_str = [f"3:PLC_m",f"3:DataBlockGlobal",f"3:CCD",f"3:send",f"3:action"]
        self.clent = client
    
    async def init_node(self) -> None:
        self.heart_node = await self.client.nodes.object.get_child(self.heart_node_str)
        self.task_id_node = await self.clent.nodes.object.get_child(self.task_id_node_str)
        self.action_node = await self.clent.nodes.object.get_child(self.action_node_str)

    async def get_heart(self):
        return await self.heart_node.read_value() 
    async def set_heart(self,val):
        return await self.heart_node.set_value(ua.DataValue(ua.Variant(val,ua.VariantType.Boolean)))
    async def get_task_id(self):
        return await self.task_id_node.read_value()
    async def get_action(self):
        return await self.action_node.read_value()

    async def set_action(self,val):
        return await self.action_node.set_value(ua.DataValue(ua.Variant(val,ua.VariantType.Int16)))

class position():
    def __init__(self,client,root_idx) -> None:
        self.heart_node_str = [f"3:PLC_m",f"3:DataBlockGlobal,",f"3:CCD",f"3:recv",f"3:heart"]
        self.task_id_node_str = [f"3:PLC_m",f"3:DataBlockGlobal",f"3:CCD",f"3:recv",f"3:task_id"]
        self.success_node_str = [f"3:PLC_m",f"3:DataBlockGlobal,",f"3:CCD",f"3:recv",f"3:success"]
        self.error_code_str = [f"3:PLC_m",f"3:DataBlockGlobal,",f"3:CCD",f"3:recv",f"3:error_code"]
        self.X_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:X"]
        self.Y_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:Y"]
        self.Z_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:Z"]
        self.S_node_str = [f"3:PLC_m", f"3:DataBlocksGlobal", f"3:CCD", f"3:recv", f"3:S"]
        self.client = client

    async def init_node(self) -> None:
        self.heart_node = await self.client.nodes.object.get_child(self.heart_node_str)
        self.task_id_node = await self.clent.nodes.object.get_child(self.task_id_node_str)
        self.success_node = await self.clent.nodes.object.get_child(self.success_node_str)
        self.error_code_node = await self.clent.nodes.object.get_child(self.error_code_str)
        self.X_node = await self.clent.nodes.object.get_child(self.X_node_str)
        self.Y_node = await self.clent.nodes.object.get_child(self.Y_node_str)
        self.Z_node = await self.clent.nodes.object.get_child(self.Z_node_str)
        self.S_node = await self.clent.nodes.object.get_child(self.S_node_str)

    async def set_heart(self,val):
        return await self.heart_node.set_value(ua.DataValue(ua.Variant(val,ua.VariantType.Boolean)))

    async def set_error_code(self,task_id_val,success_val,error_code_val) -> None:
        task_id_val = ua.DataValue(ua.Variant(task_id_val,ua.VariantType.Int16))
        success_val = ua.DataValue(ua.Variant(success_val,ua.VariantType.Boolean))
        error_code_val = ua.DataValue(ua.Variant(error_code_val,ua.VariantType.Int16))
        await self.task_id_node.set_value(task_id_val)
        await self.success_node.set_value(success_val)
        await self.error_code_node.set_value(error_code_val)
    
    async def set_position(self,task_id_val,success_val,error_code_val,X_val,Y_val,Z_val,S_val) -> None:
        task_id_val = ua.DataValue(ua.Variant(task_id_val,ua.VariantType.Int16))
        success_val = ua.DataValue(ua.Variant(success_val,ua.VariantType.Boolean))
        error_code_val = ua.DataValue(ua.Variant(error_code_val,ua.VariantType.Int16))
        X_val = ua.DataValue(ua.Variant(X_val,ua.VariantType.Float))
        Y_val = ua.DataValue(ua.Variant(Y_val,ua.VariantType.Float))
        Z_val = ua.DataValue(ua.Variant(Z_val,ua.VariantType.Float))
        S_val = ua.DataValue(ua.Variant(S_val,ua.VariantType.Float))
        await self.task_id_node.set_value(task_id_val)
        await self.success_node.set_value(success_val)
        await self.error_code_node.set_value(error_code_val)
        await self.X_node.set_value(X_val)
        await self.Y_node.set_value(Y_val)
        await self.Z_node.set_value(Z_val)
        await self.S_node.set_value(S_val)
        
class size():
    def __init__(self,client,root_idx) -> None:
        self.heart_node_str = [f"3:PLC_m",f"3:DataBlockGlobal,",f"3:CCD",f"3:recv2",f"3:size",f"3:heart"]
        self.task_id_node_str = [f"3:PLC_m",f"3:DataBlockGlobal",f"3:CCD",f"3:recv2",f"3:size",f"3:task_id"]
        self.success_node_str = [f"3:PLC_m",f"3:DataBlockGlobal,",f"3:CCD",f"3:recv2",f"3:success"]
        self.error_code_str = [f"3:PLC_m",f"3:DataBlockGlobal,",f"3:CCD",f"3:recv2",f"3:error_code"]
        self.length_node_str = [f"3:PLC_m",f"3:DataBlockGlobal,",f"3:CCD",f"3:recv2",f"3:size",f"3:length"]
        self.width_node_str = [f"3:PLC_m",f"3:DataBlockGlobal,",f"3:CCD",f"3:recv2",f"3:size",f"3:width"]
        self.hight_node_str = [f"3:PLC_m",f"3:DataBlockGlobal,",f"3:CCD",f"3:recv2",f"3:size",f"3:height"]
        self.client = client
            
    async def init_node(self):
        self.heart_node = await self.client.nodes.object.get_child(self.heart_node_str)
        self.task_id_node = await self.clent.nodes.object.get_child(self.task_id_node_str)
        self.success_node = await self.clent.nodes.object.get_child(self.success_node_str)
        self.error_code_node = await self.clent.nodes.object.get_child(self.error_code_str)
        self.length_node = await self.clent.nodes.object.get_child(self.length_node_str)
        self.width_node = await self.clent.nodes.object.get_child(self.width_node_str)
        self.hight_node = await self.clent.nodes.object.get_child(self.hight_node_str)
        
    async def set_heart(self,val):
        return await self.heart_node.set_value(ua.DataValue(ua.Variant(val,ua.VariantType.Boolean)))

    async def set_error_code(self,task_id_val,success_val,error_code_val) -> None:
        task_id_val = ua.DataValue(ua.Variant(task_id_val,ua.VariantType.Int16))
        success_val = ua.DataValue(ua.Variant(success_val,ua.VariantType.Boolean))
        error_code_val = ua.DataValue(ua.Variant(error_code_val,ua.VariantType.Int16))
        await self.task_id_node.set_value(task_id_val)
        await self.success_node.set_value(success_val)
        await self.error_code_node.set_value(error_code_val)
        
    async def set_size(self,task_id_val,success_val,error_code_val,length_val,width_val,hight_val) -> None:
        task_id_val = ua.DataValue(ua.Variant(task_id_val,ua.VariantType.Int16))
        success_val = ua.DataValue(ua.Variant(success_val,ua.VariantType.Boolean))
        error_code_val = ua.DataValue(ua.Variant(error_code_val,ua.VariantType.Int16))
        length_val = ua.DataValue(ua.Variant(length_val,ua.VariantType.Float))
        width_val = ua.DataValue(ua.Variant(width_val,ua.VariantType.Float))
        hight_val = ua.DataValue(ua.Variant(hight_val,ua.VariantType.Float))
        await self.task_id_node.set_value(task_id_val)
        await self.success_node.set_value(success_val)
        await self.error_code_node.set_value(error_code_val)
        await self.length_node.set_value(length_val)
        await self.width_node.set_value(width_val)
        await self.hight_node.set_value(hight_val)     

async def heart_set_1s(handle):
    while True:
        try:
            await handle.set_heart(False)
            await asyncio.sleep(1)
            await handle.set_heart(True)
            await asyncio.sleep(1)
        except Exception as e:
            logger.info("无法写入心跳数据, 主设备掉线")
            
async def main():
    # 创建客户端实例
    client = Client(url=url)
    
    # 异步连接服务器
    async with client:
        root_idx = await client.get_namespace_index(uri=root_uri)
        logger.info(f"Namespace index for '{root_uri}': {root_idx}")
        
        sinumerik_node = await client.nodes.objects.get_child([f"{root_idx}:Sinumerik"])
        random_value1_node = await sinumerik_node.get_child([f"{root_idx}:RandomValue1"])
        random_value2_node = await sinumerik_node.get_child([f"{root_idx}:RandomValue2"])

        random_value1 = await random_value1_node.read_value()
        random_value2 = await random_value2_node.read_value()

        logger.info(f"RandomValue1: {random_value1}")
        logger.info(f"RandomValue2: {random_value2}")

        test_node = await sinumerik_node.get_child([f"{root_idx}:Test"])
        random_value3_node = await test_node.get_child([f"{root_idx}:RandomValue3"])
        random_value3 = await random_value3_node.read_value()

        logger.info(f"RandomValue3: {random_value3}")


            
# 主函数
if __name__ == "__main__":
    # 使用当前时间生成日志文件名
    nowStr = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    logfile_name = f"logs/log_{nowStr}.txt"
    # 实例化LoggerHandler
    logger = Logger.LoggerHandler(file=logfile_name)
    asyncio.run(main())

