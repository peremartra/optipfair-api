---
title: OptiPFair Bias Visualization Tool
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false
license: apache-2.0
---

# 🔍 OptiPFair Bias Visualization Tool

Analyze potential biases in Large Language Models using advanced visualization techniques.

## 🎯 Features

- **PCA Analysis**: Visualize how model representations differ between prompt pairs in 2D space
- **Mean Difference**: Compare average activation differences across all layers  
- **Heatmap**: Detailed visualization of activation patterns in specific layers
- **Model Support**: Compatible with LLaMA, Gemma, Qwen, and custom HuggingFace models
- **Predefined Scenarios**: Ready-to-use bias testing scenarios for racial bias analysis

## 🚀 How to Use

1. **Check Backend Status**: Verify the system is ready
2. **Select Model**: Choose from predefined models or specify a custom HuggingFace model
3. **Choose Analysis Type**: Pick PCA, Mean Difference, or Heatmap visualization
4. **Configure Parameters**: Select scenarios, component types, and layer numbers
5. **Generate Visualization**: Click generate and download results

## 📚 Resources

- [OptipFair Library](https://github.com/peremartra/optipfair) - Main repository
- [Documentation](https://peremartra.github.io/optipfair/) - Official docs
- [LLM Reference Manual](https://github.com/peremartra/optipfair/blob/main/optipfair_llm_reference_manual.md) - Complete guide for using OptipFair with LLMs (ChatGPT, Claude, etc.)

## 🤖 For Developers

## 🤖 For Developers

Want to integrate OptipFair in your own projects? Check out the [LLM Reference Manual](https://github.com/peremartra/optipfair/blob/main/optipfair_llm_reference_manual.md).
- Just give the LLM Reference Manual to your favourite LLM and start working with optipfair.

Built with ❤️ using OptipFair library.