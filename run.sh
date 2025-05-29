#!/bin/bash
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=0

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR

# 导出环境变量
DTYPE="float16"

SHORT=v:,
LONG=dtype:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :
do
    case "$1" in
        # float16, float, int32
        (-v | --dtype)
            DTYPE="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

if [ ! $ASCEND_HOME_DIR ]; then
    ASCEND_HOME_DIR=/usr/local/Ascend/latest
fi
source $ASCEND_HOME_DIR/bin/setenv.bash

export DDK_PATH=$ASCEND_HOME_DIR
arch=$(uname -m)
export NPU_HOST_LIB=$ASCEND_HOME_DIR/${arch}-linux/lib64

# 测试不同输入数据类型, 修改对应代码
if [[ ${DTYPE} == "float16" ]]; then
    sed -i "s/.astype(.*)/.astype(np.float16)/g" `grep ".astype(.*)" -rl ./run/out/test_data/data/generate_data.py`
    sed -i "s/aclDataType dataType =.*;/aclDataType dataType = ACL_FLOAT16;/g" `grep "aclDataType dataType =.*;" -rl ./src/main.cpp`
    sed -i "s/dtype=.*)/dtype=np.float16)/g" `grep "dtype=.*)" -rl ./scripts/verify_result.py`
elif [[ ${DTYPE} == "float" ]]; then
    sed -i "s/.astype(.*)/.astype(np.float32)/g" `grep ".astype(.*)" -rl ./run/out/test_data/data/generate_data.py`
    sed -i "s/aclDataType dataType =.*;/aclDataType dataType = ACL_FLOAT;/g" `grep "aclDataType dataType =.*;" -rl ./src/main.cpp`
    sed -i "s/dtype=.*)/dtype=np.float32)/g" `grep "dtype=.*)" -rl ./scripts/verify_result.py`
elif [[ ${DTYPE} == "int32" ]]; then
    sed -i "s/.astype(.*)/.astype(np.int32)/g" `grep ".astype(.*)" -rl ./run/out/test_data/data/generate_data.py`
    sed -i "s/aclDataType dataType =.*;/aclDataType dataType = ACL_INT32;/g" `grep "aclDataType dataType =.*;" -rl ./src/main.cpp`
    sed -i "s/dtype=.*)/dtype=np.int32)/g" `grep "dtype=.*)" -rl ./scripts/verify_result.py`
else
    echo "ERROR: DTYPE is invalid!"
    return 1
fi

function main {
    # 1. 清除遗留生成文件和日志文件
    rm -rf $HOME/ascend/log/*
    rm -rf ./SinhCustom

    # 生成自定义算子包
    cp -r ${CURRENT_DIR}/../SinhCustom ./
    sed -i "s#/usr/local/Ascend/latest#$ASCEND_HOME_DIR#g" `grep "/usr/local/Ascend/latest" -rl ./SinhCustom/CMakePresets.json`
    cd ./SinhCustom
    rm -rf build_out
    bash build.sh
    if [ $? -ne 0 ]; then
        echo "ERROR: build custom op run package failed!"
        return 1
    fi
    echo "INFO: build custom op run package success!"
    
    # 2. 安装自定义算子包
    cd build_out
    OS_ID=$(cat /etc/os-release | grep "^ID=" | awk -F= '{print $2}')
    OS_ID=$(echo $OS_ID | sed -e 's/^"//' -e 's/"$//')
    arch=$(uname -m)
    ./custom_opp_${OS_ID}_${arch}.run --quiet
    if [ $? -ne 0 ]; then
        echo "ERROR: install custom op run package failed!"
        return 1
    fi
    echo "INFO: install custom op run package success!"

    # 3. 生成输入数据和真值数据
    cd $CURRENT_DIR/run/out/test_data/data
    python3 generate_data.py
    if [ $? -ne 0 ]; then
        echo "ERROR: generate input data failed!"
        return 1
    fi
    echo "INFO: generate input data success!"

    # 4. 编译acl可执行文件
    cd $CURRENT_DIR; rm -rf build; mkdir -p build; cd build
    cmake ../src
    if [ $? -ne 0 ]; then
        echo "ERROR: cmake failed!"
        return 1
    fi
    echo "INFO: cmake success!"
    make
    if [ $? -ne 0 ]; then
        echo "ERROR: make failed!"
        return 1
    fi
    echo "INFO: make success!"

    # 5. 运行可执行文件
    cd $CURRENT_DIR/run/out
    echo "INFO: execute op!"
    ./execute_sinh_op

    if [ $? -ne 0 ]; then
        echo "ERROR: acl executable run failed! please check your project!"
        return 1
    fi
    echo "INFO: acl executable run success!"

    # 4. 比较真值文件
    cd $CURRENT_DIR
    python3 $CURRENT_DIR/scripts/verify_result.py       \
        $CURRENT_DIR/run/out/test_data/data/input_0.bin \
        $CURRENT_DIR/run/out/result_files/output_0.bin
    if [ $? -ne 0 ]; then
        echo "ERROR: compare golden data failed! the result is wrong!"
        return 1
    fi
    echo "INFO: compare golden data success!"
}

main
