#!/bin/bash
# MPPI Baseline 实验 - 快速开始脚本

echo "======================================================================"
echo "MPPI Baseline 实验 - 快速开始"
echo "======================================================================"
echo ""
echo "本脚本将帮助你快速运行MPPI实验"
echo ""

# 检查是否在正确的目录
if [ ! -d "exp_mppi" ]; then
    echo "❌ 错误：请在项目根目录运行此脚本"
    echo "   当前目录: $(pwd)"
    echo "   应该在: /Users/caifucheng/Desktop/icml2026/flow_motion_plan"
    exit 1
fi

echo "✅ 目录检查通过"
echo ""

# 询问用户想要做什么
echo "请选择操作："
echo "  1) 测试环境"
echo "  2) 运行实验1（静态能量场）"
echo "  3) 运行实验2（目标吸引场干扰）"
echo "  4) 运行实验3（动态障碍物）"
echo "  5) 运行实验4（混合场景）"
echo "  6) 运行所有实验"
echo "  7) 查看结果"
echo "  0) 退出"
echo ""
read -p "请输入选项 [0-7]: " choice

case $choice in
    1)
        echo ""
        echo "======================================================================"
        echo "测试环境..."
        echo "======================================================================"
        python exp_mppi/scripts/test_environment.py
        ;;
    2)
        echo ""
        echo "======================================================================"
        echo "运行实验1：静态能量场"
        echo "======================================================================"
        bash exp_mppi/scripts/run_mppi_exp1.sh
        ;;
    3)
        echo ""
        echo "======================================================================"
        echo "运行实验2：目标吸引场干扰"
        echo "======================================================================"
        bash exp_mppi/scripts/run_mppi_exp2.sh
        ;;
    4)
        echo ""
        echo "======================================================================"
        echo "运行实验3：动态障碍物"
        echo "======================================================================"
        bash exp_mppi/scripts/run_mppi_exp3.sh
        ;;
    5)
        echo ""
        echo "======================================================================"
        echo "运行实验4：混合场景"
        echo "======================================================================"
        bash exp_mppi/scripts/run_mppi_exp4.sh
        ;;
    6)
        echo ""
        echo "======================================================================"
        echo "运行所有实验"
        echo "======================================================================"
        echo "⚠️  注意：这将需要较长时间（约30-60分钟）"
        read -p "确认继续？[y/N]: " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            bash exp_mppi/scripts/run_all_mppi.sh
        else
            echo "已取消"
        fi
        ;;
    7)
        echo ""
        echo "======================================================================"
        echo "查看结果"
        echo "======================================================================"
        echo ""
        echo "实验1结果："
        if [ -f "exp_mppi/outputs/exp1_static/results.json" ]; then
            echo "✅ exp_mppi/outputs/exp1_static/"
            ls -lh exp_mppi/outputs/exp1_static/
        else
            echo "❌ 未找到结果（可能还未运行）"
        fi
        echo ""
        echo "实验2结果："
        if [ -f "exp_mppi/outputs/exp2_goal/results.json" ]; then
            echo "✅ exp_mppi/outputs/exp2_goal/"
            ls -lh exp_mppi/outputs/exp2_goal/
        else
            echo "❌ 未找到结果（可能还未运行）"
        fi
        echo ""
        echo "实验3结果："
        if [ -f "exp_mppi/outputs/exp3_dynamic/results.json" ]; then
            echo "✅ exp_mppi/outputs/exp3_dynamic/"
            ls -lh exp_mppi/outputs/exp3_dynamic/
        else
            echo "❌ 未找到结果（可能还未运行）"
        fi
        echo ""
        echo "实验4结果："
        if [ -f "exp_mppi/outputs/exp4_mixed/results.json" ]; then
            echo "✅ exp_mppi/outputs/exp4_mixed/"
            ls -lh exp_mppi/outputs/exp4_mixed/
        else
            echo "❌ 未找到结果（可能还未运行）"
        fi
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo "完成！"
echo "======================================================================"
echo ""
echo "查看文档："
echo "  - 详细文档: exp_mppi/README.md"
echo "  - 快速开始: exp_mppi/QUICKSTART.md"
echo "  - 代码审查: exp_mppi/CODE_REVIEW.md"
echo "  - 最终报告: exp_mppi/FINAL_REPORT.md"
echo ""

