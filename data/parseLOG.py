# Import necessary libraries
import struct
import os
import datetime

import zipfile
import traceback
import sys

from utils.logger import logger

# 获取运行程序的绝对路径
program_abs_path = os.path.abspath(sys.argv[0])

# 获取运行程序所在的目录
program_dir = os.path.dirname(program_abs_path)

backup_dir = os.path.join(program_dir, "backup")



# Define the struct format for the device block
class EventBlockLOG1:
    def __init__(self):
        self.block_ID = 0x08
        self.Data_name = bytearray(16)
        self.Table_type = 0x00
        self.Table_number = 0x00
        self.Event_name_length = 0x2D


class SubEventName:
    def __init__(self):
        self.Second_event = bytearray(20)
        self.Elapsed_time_hhhh = bytearray(4)
        self.Elapsed_time_cccuuu = bytearray(6)
        self.Clock_time_yyyy = bytearray(4)
        self.Clock_time_cccuuu = bytearray(6)
        self.Reserved = bytearray(5)


class EventName:
    def __init__(self):
        self.First_event = bytearray(20)
        self.Elapsed_time = bytearray(6)
        self.Clock_time = bytearray(14)
        self.Block_No_of_waveform = 0x00
        self.System_event_code = 0x00
        self.Event_type = 0x00
        self.Prob = 0x00
        self.Event_page = bytearray(2)
        self.sub_event_name = SubEventName()


def sort_event_names(event_name):
    time_str1 = event_name.Clock_time.decode("utf").strip("\x00")
    time_str1 = time_str1.replace("(", "").replace(")", "")
    time1 = 0
    try:
        time1 = datetime.datetime.strptime(time_str1, "%y%m%d%H%M%S")
    except Exception as e:
        logger.info("An error occurred: " + str(e) + str(event_name.Clock_time))
    return time1


def convert_time(time_str):
    time_int = int(time_str)
    hh = time_int // 10000
    mm = (time_int % 10000) // 100
    ss = time_int % 100
    event_page = hh * 3600 + mm * 60 + ss
    event_page //= 10
    return event_page


def elapsed_time_calculate(time_str1, time_str2, time_str3):
    # 将时间字符串转换为datetime对象
    time_str3 = time_str3.decode("utf").strip("\x00")
    time3 = datetime.datetime.strptime(time_str3, "%H%M%S")
    # 计算两个时间的差值
    time_str1 = time_str1.decode("utf").strip("\x00")
    time_str2 = time_str2.decode("utf").strip("\x00")

    time_str1 = time_str1.replace("(", "").replace(")", "")
    time_str2 = time_str2.replace("(", "").replace(")", "")

    time1 = datetime.datetime.strptime(time_str1, "%y%m%d%H%M%S")
    time2 = datetime.datetime.strptime(time_str2, "%y%m%d%H%M%S")
    # 计算两个时间的差值
    time_diff = abs(time1 - time2)
    # 计算结果时间
    result_time = time3 + time_diff
    # 格式化输出结果
    elapsed_time = result_time.strftime("%H%M%S").encode("utf-8").ljust(6, b"\x00")
    hour_str = result_time.strftime("%H").zfill(4)[:4].encode()
    return hour_str, elapsed_time


def parseSubLOG(file_path):
    # Open file in binary mode
    if not os.path.exists(file_path):
        logger.info("文件不存在  " + file_path)
        return []
    try:
        with open(file_path, "r+b") as f:
            f.seek(128 + 17)
            blockCount = struct.unpack(">B", f.read(1))[0]
            # print(blockCount)
            event_names = []
            for i in range(blockCount):
                f.seek(128 + 18 + i * 20)
                data = f.read(4)
                # 将数据解析为十进制整数
                firstEventAddress = struct.unpack("<I", data)[0]

                f.seek(firstEventAddress)
                # 创建 EventBlockLOG1 对象并赋值
                event_block = EventBlockLOG1()
                event_block.block_ID = struct.unpack(">B", f.read(1))[0]
                event_block.Data_name = f.read(16)
                event_block.Table_type = struct.unpack(">B", f.read(1))[0]
                event_block.Table_number = struct.unpack(">B", f.read(1))[0]
                event_block.Event_name_length = struct.unpack(">B", f.read(1))[0]
                f.seek(128 + 18 + (22 + i) * 20)
                data = f.read(4)
                # 将数据解析为十进制整数
                subEventAddress = struct.unpack("<I", data)[0]

                f.seek(subEventAddress)
                # 创建 EventBlockLOG1 对象并赋值
                sub_event_block = EventBlockLOG1()
                sub_event_block.block_ID = struct.unpack(">B", f.read(1))[0]
                sub_event_block.Data_name = f.read(16)
                sub_event_block.Table_type = struct.unpack(">B", f.read(1))[0]
                sub_event_block.Table_number = struct.unpack(">B", f.read(1))[0]
                sub_event_block.Event_name_length = struct.unpack(">B", f.read(1))[0]
                for i in range(event_block.Table_number):
                    # 先读取第一标记事件
                    f.seek(firstEventAddress + 20 + i * 45)
                    event_name_data = f.read(45)
                    #print("event_name_data", event_name_data)
                    if not event_name_data:
                        print("break")
                        break
                    event_name = EventName()
                    event_name.First_event = event_name_data[:20]
                    event_name.Elapsed_time = event_name_data[20:26]
                    event_name.Clock_time = event_name_data[26:40]
                    (
                        event_name.Block_No_of_waveform,
                        event_name.System_event_code,
                        event_name.Event_type,
                    ) = struct.unpack(">BBB", event_name_data[40:43])
                    event_name.Event_page = event_name_data[43:45]
                    # 在读取第二事件
                    f.seek(subEventAddress + 20 + i * 45)
                    sub_event_name_data = f.read(45)
                    # 解析 EventName 对象
                    event_name.sub_event_name = SubEventName()
                    event_name.sub_event_name.Second_event = sub_event_name_data[:20]
                    event_name.sub_event_name.Elapsed_time_hhhh = sub_event_name_data[
                                                                  20:24
                                                                  ]
                    event_name.sub_event_name.Elapsed_time_cccuuu = sub_event_name_data[
                                                                    24:30
                                                                    ]
                    event_name.sub_event_name.Clock_time_yyyy = sub_event_name_data[
                                                                30:34
                                                                ]
                    event_name.sub_event_name.Clock_time_cccuuu = sub_event_name_data[
                                                                  34:40
                                                                  ]
                    event_name.sub_event_name.Reserved = sub_event_name_data[40:45]
                    event_names.append(event_name)
            return event_names
    except Exception as e:
        logger.info("An error occurred: " + str(e))
        logger.info(traceback.format_exc())
        return []


def parseLOG(file_path):
    # Open file in binary mode
    if not os.path.exists(file_path):
        logger.info("文件不存在  " + file_path)
        return []
    # os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
    backup_file(file_path)

    try:
        with open(file_path, "r+b") as f:
            f.seek(128 + 17)
            blockCount = struct.unpack(">B", f.read(1))[0]
            # print(blockCount)
            event_names = []
            for i in range(blockCount):
                f.seek(128 + 18 + i * 20)

                data = f.read(4)
                # 将数据解析为十进制整数
                offset = struct.unpack("<I", data)[0]
                # print("firstEventAddress offset ", data, offset)
                f.seek(offset)
                # 创建 EventBlockLOG1 对象并赋值
                event_block = EventBlockLOG1()
                event_block.block_ID = struct.unpack(">B", f.read(1))[0]
                event_block.Data_name = f.read(16)
                event_block.Table_type = struct.unpack(">B", f.read(1))[0]
                event_block.Table_number = struct.unpack(">B", f.read(1))[0]
                event_block.Event_name_length = struct.unpack(">B", f.read(1))[0]
                for i in range(event_block.Table_number):
                    event_name_data = f.read(45)
                    # if not event_name_data or event_name_data[0] == 0:
                    #     break
                    # 解析 EventName 对象
                    event_name = EventName()
                    event_name.First_event = event_name_data[:20]
                    event_name.Elapsed_time = event_name_data[20:26]
                    event_name.Clock_time = event_name_data[26:40]
                    # print(event_name.Clock_time)
                    (
                        event_name.Block_No_of_waveform,
                        event_name.System_event_code,
                        event_name.Event_type,
                    ) = struct.unpack(">BBB", event_name_data[40:43])
                    event_name.Event_page = event_name_data[43:45]
                    event_names.append(event_name)

            return event_names
    except Exception as e:
        logger.info("An error occurred: " + str(e))
        logger.info(traceback.format_exc())

        return []


def addLOG(file_path, addEvent_name):
    # Open file in binary mode
    if not os.path.exists(file_path):
        logger.info("文件不存在  " + file_path)
        return False
    # os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
    backup_file(file_path)
    # recover_data(file_path)
    # deleteAllAILOG(file_path)
    try:
        with open(file_path, "r+b") as f:
            f.seek(128 + 17)
            blockCount = struct.unpack(">B", f.read(1))[0]
            # print(blockCount)
            event_names = []
            for i in range(blockCount):
                f.seek(128 + 18 + i * 20)
                data = f.read(4)
                # 将数据解析为十进制整数
                firstEventAddress = struct.unpack("<I", data)[0]

                f.seek(firstEventAddress)
                # 创建 EventBlockLOG1 对象并赋值
                event_block = EventBlockLOG1()
                event_block.block_ID = struct.unpack(">B", f.read(1))[0]
                event_block.Data_name = f.read(16)
                event_block.Table_type = struct.unpack(">B", f.read(1))[0]
                event_block.Table_number = struct.unpack(">B", f.read(1))[0]
                event_block.Event_name_length = struct.unpack(">B", f.read(1))[0]
                f.seek(128 + 18 + (22 + i) * 20)
                data = f.read(4)
                # 将数据解析为十进制整数
                subEventAddress = struct.unpack("<I", data)[0]

                f.seek(subEventAddress)
                # 创建 EventBlockLOG1 对象并赋值
                sub_event_block = EventBlockLOG1()
                sub_event_block.block_ID = struct.unpack(">B", f.read(1))[0]
                sub_event_block.Data_name = f.read(16)
                sub_event_block.Table_type = struct.unpack(">B", f.read(1))[0]
                sub_event_block.Table_number = struct.unpack(">B", f.read(1))[0]
                sub_event_block.Event_name_length = struct.unpack(">B", f.read(1))[0]
                for i in range(event_block.Table_number):
                    # 先读取第一标记事件
                    f.seek(firstEventAddress + 20 + i * 45)
                    event_name_data = f.read(45)
                    #print("event_name_data", event_name_data)
                    if not event_name_data:
                        print("break")
                        break
                    event_name = EventName()
                    event_name.First_event = event_name_data[:20]
                    event_name.Elapsed_time = event_name_data[20:26]
                    event_name.Clock_time = event_name_data[26:40]
                    (
                        event_name.Block_No_of_waveform,
                        event_name.System_event_code,
                        event_name.Event_type,
                    ) = struct.unpack(">BBB", event_name_data[40:43])
                    event_name.Event_page = event_name_data[43:45]
                    # 在读取第二事件
                    f.seek(subEventAddress + 20 + i * 45)
                    sub_event_name_data = f.read(45)
                    # 解析 EventName 对象
                    event_name.sub_event_name = SubEventName()
                    event_name.sub_event_name.Second_event = sub_event_name_data[:20]
                    event_name.sub_event_name.Elapsed_time_hhhh = sub_event_name_data[
                                                                  20:24
                                                                  ]
                    event_name.sub_event_name.Elapsed_time_cccuuu = sub_event_name_data[
                                                                    24:30
                                                                    ]
                    event_name.sub_event_name.Clock_time_yyyy = sub_event_name_data[
                                                                30:34
                                                                ]
                    event_name.sub_event_name.Clock_time_cccuuu = sub_event_name_data[
                                                                  34:40
                                                                  ]
                    event_name.sub_event_name.Reserved = sub_event_name_data[40:45]
                    event_names.append(event_name)
                    # print("system")

                    # print(event_name.Clock_time)

                # # 添加新的 EventName 对象到数组
            for event_name in addEvent_name:  # 写入 排序后的EventName 对象
                # print("addEvent_name")
                # print(event_name.Clock_time)
                event_names.append(event_name)
            sorted_event_names = sorted(event_names, key=sort_event_names)
            totalCount = len(sorted_event_names)
            # print("totalCount add ", totalCount)
            blockCount = int(totalCount / 255) + 1
            if blockCount > 6:
                blockCount = 6
            if blockCount == 0:
                blockCount = 1
            f.seek(128 + 17)
            # print("blockCount adda ", blockCount)
            f.write(bytes([blockCount]))
            f.seek(128 + 17)
            blockCount = struct.unpack(">B", f.read(1))[0]
            # print("blockCount addl ", blockCount)
            last_event_name = None
            for i in range(blockCount):
                # print("i add ", i)
                f.seek(128 + 18 + i * 20)
                data = f.read(4)
                # firstEventAddress = struct.unpack("<I", data)[0]
                # if firstEventAddress == 0:
                firstEventAddress = 1024 + i * 20 + i * 255 * 45
                f.seek(128 + 18 + i * 20)
                new_firstEventAddress_bytes = firstEventAddress.to_bytes(
                    4, byteorder="little"
                )
                f.write(new_firstEventAddress_bytes)
                # print("firstEventAddress add ", firstEventAddress)
                f.seek(128 + 18 + (22 + i) * 20)
                data = f.read(4)
                subEventAddress = struct.unpack("<I", data)[0]
                if subEventAddress == 0:
                    subEventAddress = 253914 + i * 20 + i * 255 * 45
                    f.seek(128 + 18 + (22 + i) * 20)
                    new_subEventAddress_bytes = subEventAddress.to_bytes(
                        4, byteorder="little"
                    )
                    f.write(new_subEventAddress_bytes)
                # print("subEventAddress add ", subEventAddress)
                if i == (blockCount - 1):
                    num = totalCount % 255
                else:
                    num = 255
                event_block.Table_number = num
                sub_event_block.Table_number = num
                f.seek(firstEventAddress)
                # 清空数据
                # 写入event_block 对象
                f.write(bytes([event_block.block_ID]))
                f.write(event_block.Data_name)
                f.write(bytes([event_block.Table_type]))
                f.write(bytes([event_block.Table_number]))
                f.write(bytes([event_block.Event_name_length]))
                f.seek(subEventAddress)
                # 写入 sub_event_block 对象并赋值
                f.write(bytes([sub_event_block.block_ID]))
                f.write(sub_event_block.Data_name)
                f.write(bytes([sub_event_block.Table_type]))
                f.write(bytes([sub_event_block.Table_number]))
                f.write(bytes([sub_event_block.Event_name_length]))
                for i, event_name in enumerate(
                        sorted_event_names[(i * 255): (i * 255) + num]
                ):
                    # print("i=", i)
                    sub_event_name = event_name.sub_event_name
                    if (
                            config.get_event_flag()
                            in event_name.First_event.decode("gbk").strip("\x00")
                            and last_event_name
                    ):
                        (
                            sub_event_name.Elapsed_time_hhhh,
                            event_name.Elapsed_time,
                        ) = elapsed_time_calculate(
                            event_name.Clock_time,
                            last_event_name.Clock_time,
                            last_event_name.Elapsed_time,
                        )
                        event_name.Event_page = convert_time(
                            event_name.Elapsed_time
                        ).to_bytes(length=2, byteorder="little")
                        event_name.Block_No_of_waveform = (
                            last_event_name.Block_No_of_waveform
                        )
                        sub_event_name.Clock_time_yyyy = (
                            last_event_name.sub_event_name.Clock_time_yyyy
                        )
                        sub_event_name.Elapsed_time_cccuuu = (
                            last_event_name.sub_event_name.Elapsed_time_cccuuu
                        )
                        sub_event_name.Clock_time_cccuuu = (
                            last_event_name.sub_event_name.Clock_time_cccuuu
                        )
                        # sub_event_name.Elapsed_time_cccuuu = (
                        #     sub_event_name.Clock_time_cccuuu
                        # ) = "321000".encode("utf-8")
                    if config.get_event_flag() not in event_name.First_event.decode("gbk").strip("\x00"):
                        last_event_name = event_name
                    elif not last_event_name:
                        continue

                        # 写入 event_name的数据
                    f.seek(firstEventAddress + 20 + i * 45)
                    f.write(event_name.First_event)
                    f.write(event_name.Elapsed_time)
                    f.write(event_name.Clock_time)

                    f.write(
                        struct.pack(
                            ">BBB",
                            event_name.Block_No_of_waveform,
                            event_name.System_event_code,
                            event_name.Event_type,
                        )
                    )
                    f.write(event_name.Event_page)
                    print("write")
                    print(event_name.First_event)
                    print(event_name.Elapsed_time)
                    print(event_name.Clock_time)
                    print(event_name.Block_No_of_waveform)
                    print(event_name.System_event_code)
                    print(event_name.Event_type)
                    print(event_name.Event_page)

                    f.seek(subEventAddress + 20 + i * 45)
                    # 写入 sub_event_name的数据
                    f.write(sub_event_name.Second_event)
                    f.write(sub_event_name.Elapsed_time_hhhh)
                    f.write(sub_event_name.Elapsed_time_cccuuu)
                    f.write(sub_event_name.Clock_time_yyyy)
                    f.write(sub_event_name.Clock_time_cccuuu)
                    f.write(sub_event_name.Reserved)
            return True
    except Exception as e:
        logger.info("An error occurred: " + str(e))
        logger.info(traceback.format_exc())
        return False


def deleteAllAILOG(file_path):
    # Open file in binary mode
    if not os.path.exists(file_path):
        logger.info("文件不存在  " + file_path)
        return False
    # os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with open(file_path, "r+b") as f:
            f.seek(128 + 17)
            blockCount = struct.unpack(">B", f.read(1))[0]
            # print(blockCount)
            event_names = []
            for i in range(blockCount):
                f.seek(128 + 18 + i * 20)
                data = f.read(4)
                # 将数据解析为十进制整数
                firstEventAddress = struct.unpack("<I", data)[0]

                f.seek(firstEventAddress)
                # 创建 EventBlockLOG1 对象并赋值
                event_block = EventBlockLOG1()
                event_block.block_ID = struct.unpack(">B", f.read(1))[0]
                event_block.Data_name = f.read(16)
                event_block.Table_type = struct.unpack(">B", f.read(1))[0]
                event_block.Table_number = struct.unpack(">B", f.read(1))[0]
                event_block.Event_name_length = struct.unpack(">B", f.read(1))[0]
                f.seek(128 + 18 + (22 + i) * 20)
                data = f.read(4)
                # 将数据解析为十进制整数
                subEventAddress = struct.unpack("<I", data)[0]

                f.seek(subEventAddress)
                # 创建 EventBlockLOG1 对象并赋值
                sub_event_block = EventBlockLOG1()
                sub_event_block.block_ID = struct.unpack(">B", f.read(1))[0]
                sub_event_block.Data_name = f.read(16)
                sub_event_block.Table_type = struct.unpack(">B", f.read(1))[0]
                sub_event_block.Table_number = struct.unpack(">B", f.read(1))[0]
                sub_event_block.Event_name_length = struct.unpack(">B", f.read(1))[0]
                for i in range(event_block.Table_number):
                    # 先读取第一标记事件
                    f.seek(firstEventAddress + 20 + i * 45)
                    event_name_data = f.read(45)
                    if not event_name_data:
                        break
                    event_name = EventName()
                    event_name.First_event = event_name_data[:20]
                    event_name.Elapsed_time = event_name_data[20:26]
                    event_name.Clock_time = event_name_data[26:40]
                    (
                        event_name.Block_No_of_waveform,
                        event_name.System_event_code,
                        event_name.Event_type,
                    ) = struct.unpack(">BBB", event_name_data[40:43])
                    event_name.Event_page = event_name_data[43:45]
                    # 在读取第二事件
                    f.seek(subEventAddress + 20 + i * 45)
                    sub_event_name_data = f.read(45)
                    # 解析 EventName 对象
                    event_name.sub_event_name = SubEventName()
                    event_name.sub_event_name.Second_event = sub_event_name_data[:20]
                    event_name.sub_event_name.Elapsed_time_hhhh = sub_event_name_data[
                                                                  20:24
                                                                  ]
                    event_name.sub_event_name.Elapsed_time_cccuuu = sub_event_name_data[
                                                                    24:30
                                                                    ]
                    event_name.sub_event_name.Clock_time_yyyy = sub_event_name_data[
                                                                30:34
                                                                ]
                    event_name.sub_event_name.Clock_time_cccuuu = sub_event_name_data[
                                                                  34:40
                                                                  ]
                    event_name.sub_event_name.Reserved = sub_event_name_data[40:45]
                    # print(event_name.First_event)
                    # print(event_name.Clock_time)
                    try:
                        time_str1 = event_name.Clock_time.decode("utf").strip("\x00")
                        time_str1 = time_str1.replace("(", "").replace(")", "")
                        datetime.datetime.strptime(time_str1, "%y%m%d%H%M%S")
                        if config.get_event_flag() not in event_name.First_event.decode(
                                "gbk", errors="ignore"
                        ).strip("\x00"):
                            event_names.append(event_name)
                    except Exception as e:
                        logger.info(
                            "An error occurred: " + str(e) + str(event_name.Clock_time)
                        )

            sorted_event_names = sorted(event_names, key=sort_event_names)
            totalCount = len(sorted_event_names)
            # print("totalCount add ", totalCount)
            blockCount = int(totalCount / 255) + 1
            if blockCount > 6:
                blockCount = 6
            if blockCount == 0:
                blockCount = 1
            f.seek(128 + 17)
            # print("blockCount adda ", blockCount)
            f.write(bytes([blockCount]))
            f.seek(128 + 17)
            blockCount = struct.unpack(">B", f.read(1))[0]
            # print("blockCount addl ", blockCount)

            for i in range(blockCount):
                # print("i add ", i)
                f.seek(128 + 18 + i * 20)
                data = f.read(4)
                # firstEventAddress = struct.unpack("<I", data)[0]
                # if firstEventAddress == 0:
                firstEventAddress = 1024 + i * 20 + i * 255 * 45
                f.seek(128 + 18 + i * 20)
                new_firstEventAddress_bytes = firstEventAddress.to_bytes(
                    4, byteorder="little"
                )
                f.write(new_firstEventAddress_bytes)
                # print("firstEventAddress add ", firstEventAddress)
                f.seek(128 + 18 + (22 + i) * 20)
                data = f.read(4)
                subEventAddress = struct.unpack("<I", data)[0]
                if subEventAddress == 0:
                    subEventAddress = 253914 + i * 20 + i * 255 * 45
                    f.seek(128 + 18 + (22 + i) * 20)
                    new_subEventAddress_bytes = subEventAddress.to_bytes(
                        4, byteorder="little"
                    )
                    f.write(new_subEventAddress_bytes)
                # print("subEventAddress add ", subEventAddress)
                if i == (blockCount - 1):
                    num = totalCount % 255
                else:
                    num = 255
                event_block.Table_number = num
                sub_event_block.Table_number = num
                f.seek(firstEventAddress)
                # 写入event_block 对象
                f.write(bytes([event_block.block_ID]))
                f.write(event_block.Data_name)
                f.write(bytes([event_block.Table_type]))
                f.write(bytes([event_block.Table_number]))
                f.write(bytes([event_block.Event_name_length]))
                f.seek(subEventAddress)
                # 写入 sub_event_block 对象并赋值
                f.write(bytes([sub_event_block.block_ID]))
                f.write(sub_event_block.Data_name)
                f.write(bytes([sub_event_block.Table_type]))
                f.write(bytes([sub_event_block.Table_number]))
                f.write(bytes([sub_event_block.Event_name_length]))
                for i, event_name in enumerate(
                        sorted_event_names[(i * 255): (i * 255) + num]
                ):
                    sub_event_name = event_name.sub_event_name
                    # 写入 event_name的数据
                    f.seek(firstEventAddress + 20 + i * 45)
                    f.write(event_name.First_event)
                    f.write(event_name.Elapsed_time)
                    f.write(event_name.Clock_time)
                    # print("add", event_name.Clock_time)

                    f.write(
                        struct.pack(
                            ">BBB",
                            event_name.Block_No_of_waveform,
                            event_name.System_event_code,
                            event_name.Event_type,
                        )
                    )
                    f.write(event_name.Event_page)
                    f.seek(subEventAddress + 20 + i * 45)
                    # 写入 sub_event_name的数据
                    f.write(sub_event_name.Second_event)
                    f.write(sub_event_name.Elapsed_time_hhhh)
                    f.write(sub_event_name.Elapsed_time_cccuuu)
                    f.write(sub_event_name.Clock_time_yyyy)
                    f.write(sub_event_name.Clock_time_cccuuu)
                    f.write(sub_event_name.Reserved)
            return True
    except Exception as e:
        logger.info("An error occurred: " + str(e))
        logger.info(traceback.format_exc())
        return False


def backup_file(file_path):
    # 检测备份目录是否存在，如果不存在则创建目录
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # 构造备份文件名，
    file_name = os.path.basename(file_path).split(".")[0]
    # 构造备份文件路径
    backup_file_path = os.path.join(backup_dir, file_name + ".zip")

    # 检查备份文件是否已经存在，如果不存在则进行备份
    try:
        if not os.path.exists(backup_file_path):
            try:
                with zipfile.ZipFile(
                        backup_file_path, "w", zipfile.ZIP_DEFLATED
                ) as zipf:
                    # iterate through all files and directories in the given directory
                    zipf.write(file_path, os.path.basename(file_path))
                    cmt_file_path = os.path.splitext(file_path)[0] + ".CMT"
                    zipf.write(cmt_file_path, os.path.basename(cmt_file_path))

            except Exception as e:
                logger.info("An error occurred: " + str(e))
                logger.info(traceback.format_exc())
                logger.info("数据压缩失败")
                logger.info("备份文件失败：  " + file_path)
        else:
            return False
    except Exception as e:
        logger.info("An error occurred: " + str(e))
        logger.info(traceback.format_exc())
        logger.info("备份文件失败：  " + file_path)
    # 检查备份目录下的文件数量，如果超过 1000 个则删除时间最早的那个文件
    backup_files = os.listdir(backup_dir)
    if len(backup_files) > 10000:
        oldest_file = min(
            backup_files, key=lambda x: os.path.getctime(os.path.join(backup_dir, x))
        )
        os.remove(os.path.join(backup_dir, oldest_file))
    return True


def recover_data(file_path):
    # 检测备份目录是否存在，如果不存在则创建目录
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    # 构造备份文件名，格式为 "{原文件名}_{备份时间}.bak"
    file_name = os.path.basename(file_path).split(".")[0]
    backup_files = [f for f in os.listdir(backup_dir) if f.startswith(file_name)]

    # 检查备份文件是否已经存在，如果存在则使用备份文件替换现在文件
    try:
        if backup_files:
            backup_file_path = os.path.join(backup_dir, backup_files[0])

            with zipfile.ZipFile(backup_file_path, "r") as zipf:
                zipf.extractall(os.path.dirname(file_path))
            return True
        else:
            return False

    except Exception as e:
        logger.info("An error occurred: " + str(e))
        logger.info(traceback.format_exc())
        logger.info("恢复文件失败：  " + file_path)
        return False
