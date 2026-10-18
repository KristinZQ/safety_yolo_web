const $ = (selector) => document.querySelector(selector);
const HIT_TOLERANCE = 6;
const IMAGE_EXTENSIONS = new Set(["jpg", "jpeg", "png", "bmp", "webp"]);

const state = {
  currentDatasetId: null,
  imageList: [],
  currentImageIndex: -1,
  currentImage: null,
  classes: [],
  selectedClass: null,
  boxes: [],
  historyStack: [],
  isDrawing: false,
  startPoint: null,
  previewBox: null,
  isDirty: false,
  imageElement: new Image(),
  displayWidth: 0,
  displayHeight: 0,
  originalWidth: 0,
  originalHeight: 0,
  scaleX: 1,
  scaleY: 1,
};

const canvas = $("#annotationCanvas");
const ctx = canvas.getContext("2d");

function uuid() {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return `box_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function setStatus(text, isError = false) {
  $("#statusText").textContent = text || "";
  $("#statusText").style.color = isError ? "#c23a2b" : "#1f7a4d";
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof body === "object" ? body.detail || body.message || JSON.stringify(body) : body;
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return body;
}

function extOf(file) {
  return (file.name.split(".").pop() || "").toLowerCase();
}

function yoloFromBox(box) {
  const x1 = Math.max(0, Math.min(box.x1, box.x2));
  const x2 = Math.min(state.originalWidth, Math.max(box.x1, box.x2));
  const y1 = Math.max(0, Math.min(box.y1, box.y2));
  const y2 = Math.min(state.originalHeight, Math.max(box.y1, box.y2));
  const width = Math.max(0, x2 - x1);
  const height = Math.max(0, y2 - y1);
  const xCenter = ((x1 + x2) / 2) / state.originalWidth;
  const yCenter = ((y1 + y2) / 2) / state.originalHeight;
  return [box.class_id, xCenter, yCenter, width / state.originalWidth, height / state.originalHeight]
    .map((value, index) => (index === 0 ? String(value) : Number(value).toFixed(6)))
    .join(" ");
}

async function loadClasses() {
  try {
    const data = await apiJson("/api/annotation/classes");
    state.classes = data.classes || [];
    state.selectedClass = state.classes[0] || null;
    renderClasses();
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function loadHealthStatus() {
  try {
    const health = await apiJson("/api/health");
    $("#serviceStatus").textContent = health.server || health.status;
    $("#modelStatus").textContent = health.model_loaded ? "已加载" : "未加载";
  } catch (error) {
    $("#serviceStatus").textContent = "异常";
    $("#modelStatus").textContent = "未加载";
    setStatus(error.message, true);
  }
}

async function parseYamlClasses(file) {
  if (!file) return;
  const form = new FormData();
  form.append("yaml_file", file);
  try {
    const data = await apiJson("/api/models/parse-yaml", { method: "POST", body: form });
    const palette = ["#00AEEF", "#00CC66", "#FF9900", "#9933FF", "#FF3333", "#4D96FF", "#F46036", "#2EC4B6"];
    state.classes = (data.classes || []).map((item, index) => ({
      id: Number(item.id),
      name: item.name,
      color: palette[index % palette.length],
    }));
    state.selectedClass = state.classes[0] || null;
    renderClasses();
    setStatus(data.is_contiguous ? "已根据 YAML 生成类别" : "已生成类别，注意 YAML class_id 不连续");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function renderClasses() {
  renderClassEditor();
  renderClassButtons();
}

function renderClassEditor() {
  const editor = $("#classEditor");
  editor.innerHTML = state.classes
    .map(
      (item, index) => `
        <div class="class-row" data-index="${index}">
          <input class="class-id" type="number" min="0" value="${item.id}" />
          <input class="class-name" type="text" value="${item.name}" />
          <input class="class-color" type="color" value="${item.color}" />
          <button class="delete-class" type="button" title="删除类别">×</button>
        </div>
      `
    )
    .join("");
}

function renderClassButtons() {
  const buttons = $("#classButtons");
  buttons.innerHTML = state.classes
    .map((item) => {
      const active = state.selectedClass && state.selectedClass.id === item.id ? "active" : "";
      return `<button type="button" class="class-button ${active}" data-class-id="${item.id}" style="border-color:${item.color};${active ? `background:${item.color};` : ""}">${item.id}: ${item.name}</button>`;
    })
    .join("");
}

function readClassesFromEditor() {
  return Array.from(document.querySelectorAll(".class-row")).map((row) => ({
    id: Number(row.querySelector(".class-id").value),
    name: row.querySelector(".class-name").value.trim(),
    color: row.querySelector(".class-color").value,
  }));
}

async function saveClasses() {
  try {
    const classes = readClassesFromEditor();
    const data = await apiJson("/api/annotation/classes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ classes }),
    });
    state.classes = data.classes;
    state.selectedClass = state.classes.find((item) => state.selectedClass && item.id === state.selectedClass.id) || state.classes[0] || null;
    renderClasses();
    setStatus("类别配置已保存");
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList).filter((file) => IMAGE_EXTENSIONS.has(extOf(file)));
  if (!files.length) {
    setStatus("没有可上传的图片文件", true);
    return;
  }
  if (!(await confirmDiscardDirty())) return;
  const form = new FormData();
  files.forEach((file) => {
    form.append("files", file);
    form.append("relative_paths", file.webkitRelativePath || file.name);
  });
  setStatus("图片上传中");
  try {
    const data = await apiJson("/api/annotation/upload", { method: "POST", body: form });
    state.currentDatasetId = data.dataset_id;
    state.imageList = data.images || [];
    localStorage.setItem("annotationDatasetId", state.currentDatasetId);
    renderImageSelect();
    await loadImageAt(0, true);
    $("#downloadAllBtn").classList.remove("disabled");
    $("#downloadAllBtn").href = `/api/annotation/datasets/${state.currentDatasetId}/labels.zip`;
    setStatus(`已上传 ${data.total} 张图片`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function loadSavedDataset() {
  const datasetId = localStorage.getItem("annotationDatasetId");
  if (!datasetId) return;
  try {
    const data = await apiJson(`/api/annotation/datasets/${datasetId}`);
    state.currentDatasetId = data.dataset_id;
    state.imageList = data.images || [];
    renderImageSelect();
    $("#downloadAllBtn").classList.remove("disabled");
    $("#downloadAllBtn").href = `/api/annotation/datasets/${state.currentDatasetId}/labels.zip`;
    if (state.imageList.length) {
      await loadImageAt(0, true);
    }
  } catch (_) {
    localStorage.removeItem("annotationDatasetId");
  }
}

function renderImageSelect() {
  $("#imageSelect").innerHTML = state.imageList
    .map((image, index) => {
      const label = `${index + 1}. ${image.labeled ? "已标注" : "未标注"} - ${image.relative_path || image.original_filename || image.filename}`;
      return `<option value="${index}">${label}</option>`;
    })
    .join("");
}

async function confirmDiscardDirty() {
  if (!state.isDirty) return true;
  return confirm("当前图片还有未保存修改，确认放弃这些修改吗？");
}

async function loadImageAt(index, force = false) {
  if (index < 0 || index >= state.imageList.length) return;
  if (!force && !(await confirmDiscardDirty())) return;
  state.currentImageIndex = index;
  state.currentImage = state.imageList[index];
  $("#imageSelect").value = String(index);
  state.historyStack = [];
  state.isDirty = false;
  try {
    const data = await apiJson(
      `/api/annotation/datasets/${state.currentDatasetId}/images/${state.currentImage.image_id}/annotations`
    );
    state.boxes = data.boxes || [];
  } catch (error) {
    state.boxes = [];
    setStatus(error.message, true);
  }

  await loadImageElement(state.currentImage.url);
  $("#downloadLabelBtn").href = `/api/annotation/datasets/${state.currentDatasetId}/images/${state.currentImage.image_id}/label`;
  $("#downloadLabelBtn").setAttribute("download", state.currentImage.label_filename || `${state.currentImage.original_stem || "label"}.txt`);
  $("#downloadLabelBtn").classList.toggle("disabled", !state.currentImage.labeled);
  updateAllViews();
}

function loadImageElement(url) {
  return new Promise((resolve, reject) => {
    state.imageElement = new Image();
    state.imageElement.onload = () => {
      state.originalWidth = state.imageElement.naturalWidth || state.currentImage.width;
      state.originalHeight = state.imageElement.naturalHeight || state.currentImage.height;
      resizeCanvasToImage();
      $("#emptyHint").style.display = "none";
      resolve();
    };
    state.imageElement.onerror = () => reject(new Error("图片加载失败"));
    state.imageElement.src = `${url}?t=${Date.now()}`;
  });
}

function resizeCanvasToImage() {
  if (!state.imageElement.naturalWidth) return;
  const wrap = $("#canvasWrap");
  const maxWidth = Math.max(320, wrap.clientWidth - 28);
  const maxHeight = Math.max(320, window.innerHeight - 170);
  const scale = Math.min(maxWidth / state.originalWidth, maxHeight / state.originalHeight, 1);
  state.displayWidth = Math.max(1, Math.round(state.originalWidth * scale));
  state.displayHeight = Math.max(1, Math.round(state.originalHeight * scale));
  state.scaleX = state.originalWidth / state.displayWidth;
  state.scaleY = state.originalHeight / state.displayHeight;
  canvas.width = state.displayWidth;
  canvas.height = state.displayHeight;
  canvas.style.width = `${state.displayWidth}px`;
  canvas.style.height = `${state.displayHeight}px`;
  drawCanvas();
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) * canvas.width) / rect.width,
    y: ((event.clientY - rect.top) * canvas.height) / rect.height,
  };
}

function displayToOriginal(point) {
  return {
    x: Math.max(0, Math.min(state.originalWidth, point.x * state.scaleX)),
    y: Math.max(0, Math.min(state.originalHeight, point.y * state.scaleY)),
  };
}

function originalToDisplayBox(box) {
  return {
    x1: box.x1 / state.scaleX,
    y1: box.y1 / state.scaleY,
    x2: box.x2 / state.scaleX,
    y2: box.y2 / state.scaleY,
  };
}

function drawCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.imageElement.naturalWidth) return;
  ctx.drawImage(state.imageElement, 0, 0, state.displayWidth, state.displayHeight);
  state.boxes.forEach(drawBox);
  if (state.previewBox) {
    drawDisplayBox(state.previewBox, state.selectedClass ? state.selectedClass.color : "#FFFFFF", "preview", true);
  }
}

function drawBox(box) {
  const displayBox = originalToDisplayBox(box);
  drawDisplayBox(displayBox, box.color, `${box.class_id}: ${box.class_name}`, false);
}

function drawDisplayBox(box, color, label, dashed) {
  const x = Math.min(box.x1, box.x2);
  const y = Math.min(box.y1, box.y2);
  const width = Math.abs(box.x2 - box.x1);
  const height = Math.abs(box.y2 - box.y1);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.setLineDash(dashed ? [6, 4] : []);
  ctx.strokeRect(x, y, width, height);
  ctx.setLineDash([]);
  ctx.font = "13px Segoe UI, Arial";
  const labelWidth = ctx.measureText(label).width + 10;
  ctx.fillStyle = color;
  ctx.fillRect(x, Math.max(0, y - 22), labelWidth, 22);
  ctx.fillStyle = "#FFFFFF";
  ctx.fillText(label, x + 5, Math.max(14, y - 7));
  ctx.restore();
}

function startDrawing(event) {
  if (event.button !== 0 || !state.currentImage) return;
  if (!state.selectedClass) {
    setStatus("请先选择类别", true);
    return;
  }
  state.isDrawing = true;
  state.startPoint = canvasPoint(event);
  state.previewBox = { x1: state.startPoint.x, y1: state.startPoint.y, x2: state.startPoint.x, y2: state.startPoint.y };
}

function moveDrawing(event) {
  if (!state.isDrawing || !state.startPoint) return;
  const point = canvasPoint(event);
  state.previewBox = { x1: state.startPoint.x, y1: state.startPoint.y, x2: point.x, y2: point.y };
  drawCanvas();
}

function finishDrawing(event) {
  if (!state.isDrawing || !state.startPoint) return;
  const endPoint = canvasPoint(event);
  const displayWidth = Math.abs(endPoint.x - state.startPoint.x);
  const displayHeight = Math.abs(endPoint.y - state.startPoint.y);
  state.isDrawing = false;
  state.previewBox = null;
  if (displayWidth < 3 || displayHeight < 3) {
    drawCanvas();
    return;
  }
  const originalStart = displayToOriginal(state.startPoint);
  const originalEnd = displayToOriginal(endPoint);
  const box = {
    id: uuid(),
    class_id: state.selectedClass.id,
    class_name: state.selectedClass.name,
    color: state.selectedClass.color,
    x1: Math.round(Math.min(originalStart.x, originalEnd.x) * 100) / 100,
    y1: Math.round(Math.min(originalStart.y, originalEnd.y) * 100) / 100,
    x2: Math.round(Math.max(originalStart.x, originalEnd.x) * 100) / 100,
    y2: Math.round(Math.max(originalStart.y, originalEnd.y) * 100) / 100,
  };
  state.boxes.push(box);
  state.historyStack.push({ type: "add", box });
  state.isDirty = true;
  updateAllViews();
}

function hitTestBoxBorder(mouseX, mouseY, box) {
  const displayBox = originalToDisplayBox(box);
  const left = Math.min(displayBox.x1, displayBox.x2);
  const right = Math.max(displayBox.x1, displayBox.x2);
  const top = Math.min(displayBox.y1, displayBox.y2);
  const bottom = Math.max(displayBox.y1, displayBox.y2);
  const insideX = mouseX >= left - HIT_TOLERANCE && mouseX <= right + HIT_TOLERANCE;
  const insideY = mouseY >= top - HIT_TOLERANCE && mouseY <= bottom + HIT_TOLERANCE;
  if (!insideX || !insideY) return false;
  const distances = [];
  if (mouseY >= top - HIT_TOLERANCE && mouseY <= bottom + HIT_TOLERANCE) {
    distances.push(Math.abs(mouseX - left), Math.abs(mouseX - right));
  }
  if (mouseX >= left - HIT_TOLERANCE && mouseX <= right + HIT_TOLERANCE) {
    distances.push(Math.abs(mouseY - top), Math.abs(mouseY - bottom));
  }
  if (!distances.length) return false;
  return Math.min(...distances) <= HIT_TOLERANCE;
}

function deleteBoxByRightClick(event) {
  event.preventDefault();
  if (!state.currentImage) return;
  const point = canvasPoint(event);
  const candidates = state.boxes
    .map((box, index) => ({ box, index, area: Math.abs((box.x2 - box.x1) * (box.y2 - box.y1)) }))
    .filter((item) => hitTestBoxBorder(point.x, point.y, item.box))
    .sort((a, b) => a.area - b.area || b.index - a.index);
  if (!candidates.length) return;
  const target = candidates[0];
  state.boxes.splice(target.index, 1);
  state.historyStack.push({ type: "delete", box: target.box, index: target.index });
  state.isDirty = true;
  updateAllViews();
}

function undo() {
  const action = state.historyStack.pop();
  if (!action) return;
  if (action.type === "add") {
    state.boxes = state.boxes.filter((box) => box.id !== action.box.id);
  }
  if (action.type === "delete") {
    state.boxes.splice(action.index, 0, action.box);
  }
  if (action.type === "clear") {
    state.boxes = action.boxes;
  }
  state.isDirty = true;
  updateAllViews();
}

function clearBoxes() {
  if (!state.boxes.length) return;
  state.historyStack.push({ type: "clear", boxes: [...state.boxes] });
  state.boxes = [];
  state.isDirty = true;
  updateAllViews();
}

async function saveCurrentAnnotations() {
  if (!state.currentDatasetId || !state.currentImage) {
    setStatus("请先上传图片", true);
    return;
  }
  try {
    const result = await apiJson(
      `/api/annotation/datasets/${state.currentDatasetId}/images/${state.currentImage.image_id}/annotations`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image_width: state.originalWidth,
          image_height: state.originalHeight,
          boxes: state.boxes,
        }),
      }
    );
    state.isDirty = false;
    state.currentImage.labeled = true;
    state.currentImage.label_url = result.label_url;
    state.currentImage.label_filename = result.label_filename || state.currentImage.label_filename;
    $("#downloadLabelBtn").href = result.label_url;
    $("#downloadLabelBtn").setAttribute("download", state.currentImage.label_filename || "");
    $("#downloadLabelBtn").classList.remove("disabled");
    renderImageSelect();
    setStatus("标签已保存");
    if (result.next_image_id) {
      const nextIndex = state.imageList.findIndex((image) => image.image_id === result.next_image_id);
      if (nextIndex >= 0) {
        await loadImageAt(nextIndex, true);
      }
    }
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function clearServerAnnotations() {
  clearBoxes();
  if (!state.currentDatasetId || !state.currentImage) return;
  try {
    await apiJson(`/api/annotation/datasets/${state.currentDatasetId}/images/${state.currentImage.image_id}/annotations`, {
      method: "DELETE",
    });
    state.currentImage.labeled = false;
    $("#downloadLabelBtn").classList.add("disabled");
    renderImageSelect();
    setStatus("当前图片标注已清空");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function updateAllViews() {
  drawCanvas();
  renderInfo();
  renderBoxList();
}

function renderInfo() {
  const image = state.currentImage;
  const total = state.imageList.length;
  const labeled = state.imageList.filter((item) => item.labeled).length;
  $("#progressText").textContent = image ? `第 ${state.currentImageIndex + 1} / ${total} 张，已标注 ${state.boxes.length} 个目标` : "未加载图片";
  $("#infoFilename").textContent = image ? image.original_filename || image.filename : "-";
  $("#infoSize").textContent = image ? `${state.originalWidth} × ${state.originalHeight}` : "-";
  $("#infoBoxCount").textContent = String(state.boxes.length);
  $("#infoProgress").textContent = total ? `${labeled} / ${total}` : "-";
}

function renderBoxList() {
  if (!state.boxes.length) {
    $("#boxList").innerHTML = `<div class="box-card">暂无标注框</div>`;
    return;
  }
  $("#boxList").innerHTML = state.boxes
    .map((box, index) => {
      const xyxy = `${box.x1.toFixed(1)}, ${box.y1.toFixed(1)}, ${box.x2.toFixed(1)}, ${box.y2.toFixed(1)}`;
      return `
        <div class="box-card" style="border-left-color:${box.color}">
          <strong>${index + 1}. ${box.class_id}: ${box.class_name}</strong>
          <code>xyxy: ${xyxy}</code>
          <code>YOLO: ${yoloFromBox(box)}</code>
        </div>
      `;
    })
    .join("");
}

function bindEvents() {
  $("#singleImageInput").addEventListener("change", (event) => uploadFiles(event.target.files));
  $("#folderInput").addEventListener("change", (event) => uploadFiles(event.target.files));
  $("#classYamlInput").addEventListener("change", (event) => parseYamlClasses(event.target.files[0]));
  $("#imageSelect").addEventListener("change", (event) => loadImageAt(Number(event.target.value)));
  $("#addClassBtn").addEventListener("click", () => {
    const maxId = state.classes.reduce((max, item) => Math.max(max, Number(item.id)), -1);
    state.classes.push({ id: maxId + 1, name: `class_${maxId + 1}`, color: "#4D96FF" });
    renderClasses();
  });
  $("#classEditor").addEventListener("input", () => {
    state.classes = readClassesFromEditor();
    if (state.selectedClass) {
      state.selectedClass = state.classes.find((item) => item.id === state.selectedClass.id) || state.classes[0] || null;
    }
    renderClassButtons();
  });
  $("#classEditor").addEventListener("click", (event) => {
    if (!event.target.classList.contains("delete-class")) return;
    const row = event.target.closest(".class-row");
    state.classes.splice(Number(row.dataset.index), 1);
    state.selectedClass = state.classes[0] || null;
    renderClasses();
  });
  $("#classButtons").addEventListener("click", (event) => {
    const button = event.target.closest(".class-button");
    if (!button) return;
    state.selectedClass = state.classes.find((item) => item.id === Number(button.dataset.classId)) || null;
    renderClasses();
  });
  $("#saveClassesBtn").addEventListener("click", saveClasses);
  $("#modelClassesBtn").addEventListener("click", async () => {
    try {
      const data = await apiJson("/api/annotation/classes/model");
      state.classes = data.classes || [];
      state.selectedClass = state.classes[0] || null;
      renderClasses();
      setStatus("已读取模型类别，确认后请保存类别配置");
    } catch (error) {
      setStatus(error.message, true);
    }
  });
  $("#saveLabelBtn").addEventListener("click", saveCurrentAnnotations);
  $("#undoBtn").addEventListener("click", undo);
  $("#clearBtn").addEventListener("click", clearServerAnnotations);
  $("#prevBtn").addEventListener("click", () => loadImageAt(state.currentImageIndex - 1));
  $("#nextBtn").addEventListener("click", () => loadImageAt(state.currentImageIndex + 1));
  $("#downloadLabelBtn").addEventListener("click", (event) => {
    if (state.isDirty) {
      event.preventDefault();
      setStatus("当前图片有未保存修改，请先保存标签", true);
    }
  });
  canvas.addEventListener("mousedown", startDrawing);
  canvas.addEventListener("mousemove", moveDrawing);
  canvas.addEventListener("mouseup", finishDrawing);
  canvas.addEventListener("mouseleave", () => {
    if (state.isDrawing) {
      state.isDrawing = false;
      state.previewBox = null;
      drawCanvas();
    }
  });
  canvas.addEventListener("contextmenu", deleteBoxByRightClick);
  window.addEventListener("resize", resizeCanvasToImage);
  window.addEventListener("beforeunload", (event) => {
    if (!state.isDirty) return;
    event.preventDefault();
    event.returnValue = "";
  });
}

bindEvents();
loadHealthStatus();
loadClasses();
loadSavedDataset();
