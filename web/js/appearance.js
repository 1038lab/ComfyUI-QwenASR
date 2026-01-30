import { app } from "/scripts/app.js";

const COLOR_THEMES = {
    QwenASR: { nodeColor: "#151c19", nodeBgColor: "#444d49", width: 340 },
};

const NODE_COLORS = {
    // QwenTTS nodes
    "AILab_Qwen3ASR": "QwenASR",
    "AILab_Qwen3ASRSubtitle": "QwenASR",
};

function setNodeColors(node, theme) {
    if (!theme) { return; }
    if (theme.nodeColor) {
        node.color = theme.nodeColor;
    }
    if (theme.nodeBgColor) {
        node.bgcolor = theme.nodeBgColor;
    }
    if (theme.width) {
        node.size = node.size || [140, 80];
        node.size[0] = theme.width;
    }
}

const ext = {
    name: "QwenASR.appearance",

    nodeCreated(node) {
        const nclass = node.comfyClass;
        if (NODE_COLORS.hasOwnProperty(nclass)) {
            let colorKey = NODE_COLORS[nclass];
            const theme = COLOR_THEMES[colorKey];
            setNodeColors(node, theme);
        }
    }
};

app.registerExtension(ext);