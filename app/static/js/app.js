const $ = (selector) => document.querySelector(selector);

const state = {
  imageResults: [],
  streamId: null,
  socket: null,
  videoPollTimer: null,
  models: [],
  activeModelId: null,
  selectedModelId: null,
};

function params() {
  return {
    conf: $("#confInput").value || "0.25",
    iou: $("#iouInput").value || "0.45",
    imgsz: $("#imgszInput").value || "640",
  };
}

function setMessage(text, isError = false) {
  const message = $("#message");
  message.textContent = text || "";
  message.style.color = isError ? "#b33b26" : "#1f7a4d";
}

function withCacheBust(url) {
  if (!url) return "";
  const sep = url.includes("?") ? "&" : "?";
  return `${url}${sep}t=${Date.now()}`;
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

async function clearDetectionCache() {
  try {
    await apiJson("/api/cache/detection", { method: "POST" });
  } catch (error) {
    setMessage(`\u7f13\u5b58\u6e05\u7406\u5931\u8d25\uff1a${error.message}`, true);
  }
}

function resetImageUploadView() {
  state.imageResults = [];
  $("#sourceImage").removeAttribute("src");
  $("#resultImage").removeAttribute("src");
  $("#imageResultSelect").innerHTML = "";
  $("#imageDownload").removeAttribute("href");
  $("#imageDownload").classList.add("disabled");
  renderStats($("#imageStats"), {});
  renderDetections($("#imageDetections"), []);
}

function resetVideoUploadView() {
  clearInterval(state.videoPollTimer);
  state.videoPollTimer = null;
  $("#videoProgress").value = 0;
  $("#videoTaskStatus").textContent = "\u672a\u5f00\u59cb";
  $("#resultVideo").pause();
  $("#resultVideo").removeAttribute("src");
  $("#resultVideo").load();
  $("#videoDownload").removeAttribute("href");
  $("#videoDownload").classList.add("disabled");
  renderVideoSummary({});
}

async function prepareDetectionUpload(kind) {
  if (kind === "image") {
    resetImageUploadView();
  }
  if (kind === "video") {
    resetVideoUploadView();
  }
  await clearDetectionCache();
}

function renderStats(target, stats = {}) {
  const items = stats.report_cards
    ? stats.report_cards.map((item) => [item.label, item.value, item.tone])
    :
    stats.total_detections !== undefined
      ? [
          ["总目标数", stats.total_detections ?? 0],
          ["违规目标数", stats.violation_total ?? 0, "warning"],
          ["平均置信度", stats.avg_confidence ?? 0],
          ["违规统计", formatCounts(stats.violation_counts) || "-"],
          ["FPS", stats.fps ?? 0],
          ["耗时 ms", stats.inference_ms ?? 0],
        ]
      : [
          ["总人数", stats.person_count ?? 0],
          ["已穿安全服", stats.vest_worn_count ?? 0],
          ["未穿安全服", stats.no_vest_count ?? 0, "warning"],
          ["未戴安全帽", stats.no_helmet_count ?? 0, "warning"],
          ["平均违规置信度", stats.avg_violation_confidence ?? 0],
          ["FPS", stats.fps ?? 0],
          ["耗时 ms", stats.inference_ms ?? 0],
        ];
  target.innerHTML = items
    .map(
      ([label, value, tone]) => `
        <div class="stat-card ${tone || ""}">
          <span>${label}</span>
          <strong>${value}</strong>
        </div>
      `
    )
    .join("");
}

function formatCounts(counts = {}) {
  return Object.entries(counts)
    .map(([name, count]) => `${name}:${count}`)
    .join(" / ");
}

function renderDetections(target, detections = []) {
  if (!detections.length) {
    target.innerHTML = `<tr><td colspan="5">暂无检测框</td></tr>`;
    return;
  }
  target.innerHTML = detections
    .map((det) => {
      const bbox = (det.bbox || []).map((value) => Number(value).toFixed(1)).join(", ");
      const violation = det.is_violation
        ? `<span class="pill bad">${det.violation_type || "violation"}</span>`
        : `<span class="pill">否</span>`;
      return `
        <tr>
          <td>${det.id}</td>
          <td>${det.class_name}</td>
          <td>${Number(det.confidence || 0).toFixed(4)}</td>
          <td>${bbox}</td>
          <td>${violation}</td>
        </tr>
      `;
    })
    .join("");
}

function renderImageResult(index) {
  const result = state.imageResults[index];
  if (!result) return;
  $("#sourceImage").src = withCacheBust(result.original_image_url);
  $("#resultImage").src = withCacheBust(result.result_image_url);
  $("#imageDownload").href = result.result_image_url;
  $("#imageDownload").classList.remove("disabled");
  renderStats($("#imageStats"), result.stats);
  renderDetections($("#imageDetections"), result.detections);
}

async function loadHealth() {
  try {
    const health = await apiJson("/api/health");
    $("#serviceStatus").textContent = health.server || health.status;
    $("#modelStatus").textContent = health.model_loaded ? "已加载" : "未加载";
    if (health.active_model) {
      $("#activeModelText").textContent = `当前模型：${health.active_model.display_name}（${health.active_model.model_type}）`;
    }
    $("#confInput").value = health.default_conf;
    $("#confRange").value = health.default_conf;
    $("#iouInput").value = health.default_iou;
    $("#imgszInput").value = health.default_imgsz;
    const classNames = Object.values(health.model_classes || {});
    const modelClasses = $("#modelClasses");
    if (modelClasses) {
      modelClasses.textContent = classNames.length ? `模型类别：${classNames.join("、")}` : "模型类别：未读取到类别";
    }
  } catch (error) {
    $("#serviceStatus").textContent = "异常";
    $("#modelStatus").textContent = "未加载";
    const modelClasses = $("#modelClasses");
    if (modelClasses) {
      modelClasses.textContent = "模型类别：服务未启动或模型未加载";
    }
    setMessage(error.message, true);
  }
}

async function loadModels(selectModelId = null) {
  try {
    const data = await apiJson("/api/models");
    state.models = data.models || [];
    state.activeModelId = data.active_model_id;
    state.selectedModelId = selectModelId || state.selectedModelId || state.activeModelId || (state.models[0] && state.models[0].model_id);
    renderModelSelect();
    renderSelectedModel();
  } catch (error) {
    setMessage(error.message, true);
  }
}

function renderModelSelect() {
  const select = $("#modelSelect");
  select.innerHTML = state.models
    .map((model) => {
      const active = model.model_id === state.activeModelId ? "当前" : "可选";
      const suffix = model.is_default ? "默认" : active;
      return `<option value="${model.model_id}">${model.display_name} - ${suffix}</option>`;
    })
    .join("");
  if (state.selectedModelId) select.value = state.selectedModelId;
}

function selectedModel() {
  return state.models.find((model) => model.model_id === state.selectedModelId) || null;
}

function renderSelectedModel() {
  const model = selectedModel();
  if (!model) {
    $("#mappingTable").innerHTML = `<tr><td colspan="6">暂无模型</td></tr>`;
    return;
  }
  $("#modelDisplayName").value = model.display_name || "";
  $("#modelTypeSelect").value = model.model_type || "custom";
  $("#activeModelText").textContent = `当前模型：${(state.models.find((item) => item.model_id === state.activeModelId) || {}).display_name || "未加载"}`;
  const mapping = model.label_mapping || {};
  const names = model.names || {};
  $("#mappingTable").innerHTML = Object.entries(names)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([classId, label]) => {
      const item = mapping[label] || {};
      return `
        <tr data-label="${label}">
          <td>${classId}</td>
          <td>${label}</td>
          <td><input class="mapping-business" type="text" value="${item.business_label || label}" /></td>
          <td><input class="mapping-report" type="text" value="${item.report_label || item.business_label || label}" /></td>
          <td><input class="mapping-violation" type="checkbox" ${item.is_violation ? "checked" : ""} /></td>
          <td><input class="mapping-violation-type" type="text" value="${item.violation_type || ""}" placeholder="no_vest / custom_violation" /></td>
        </tr>
      `;
    })
    .join("");
}

function readMappingFromTable() {
  const mapping = {};
  document.querySelectorAll("#mappingTable tr[data-label]").forEach((row) => {
    const label = row.dataset.label;
    mapping[label] = {
      business_label: row.querySelector(".mapping-business").value.trim() || label,
      display_label: row.querySelector(".mapping-business").value.trim() || label,
      report_label: row.querySelector(".mapping-report").value.trim() || row.querySelector(".mapping-business").value.trim() || label,
      is_violation: row.querySelector(".mapping-violation").checked,
      violation_type: row.querySelector(".mapping-violation-type").value.trim() || null,
    };
  });
  return mapping;
}

async function uploadModel() {
  const weight = $("#modelWeightFile").files[0];
  const yaml = $("#modelYamlFile").files[0];
  if (!weight && !yaml) {
    setMessage("请上传模型权重或 YAML 文件", true);
    return;
  }
  const form = new FormData();
  if (weight) form.append("weight_file", weight);
  if (yaml) form.append("yaml_file", yaml);
  form.append("display_name", $("#modelDisplayName").value.trim());
  form.append("model_type", $("#modelTypeSelect").value);
  try {
    const data = await apiJson("/api/models/upload", { method: "POST", body: form });
    state.selectedModelId = data.model.model_id;
    await loadModels(data.model.model_id);
    setMessage("模型已上传并解析类别，请确认标签映射后保存");
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function saveMapping() {
  const model = selectedModel();
  if (!model) return;
  try {
    await apiJson(`/api/models/${model.model_id}/mapping`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        display_name: $("#modelDisplayName").value.trim() || model.display_name,
        model_type: $("#modelTypeSelect").value,
        label_mapping: readMappingFromTable(),
      }),
    });
    await loadModels(model.model_id);
    setMessage("标签业务映射已保存");
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function activateModel() {
  const model = selectedModel();
  if (!model) return;
  try {
    await apiJson(`/api/models/${model.model_id}/activate`, { method: "POST" });
    await loadHealth();
    await loadModels(model.model_id);
    setMessage("模型已激活");
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function deleteModel() {
  const model = selectedModel();
  if (!model) return;
  if (!confirm(`确认删除模型：${model.display_name}？`)) return;
  try {
    await apiJson(`/api/models/${model.model_id}`, { method: "DELETE" });
    state.selectedModelId = null;
    await loadModels();
    setMessage("模型已删除");
  } catch (error) {
    setMessage(error.message, true);
  }
}

async function detectImages() {
  const input = $("#imageFile");
  if (!input.files.length) {
    setMessage("请选择图片文件", true);
    return;
  }
  await prepareDetectionUpload("image");
  const form = new FormData();
  const selected = Array.from(input.files);
  const p = params();
  selected.forEach((file) => form.append(selected.length > 1 ? "files" : "file", file));
  form.append("conf", p.conf);
  form.append("iou", p.iou);
  form.append("imgsz", p.imgsz);

  setMessage("图片检测中");
  $("#detectImageBtn").disabled = true;
  try {
    const url = selected.length > 1 ? "/api/detect/images" : "/api/detect/image";
    const data = await apiJson(url, { method: "POST", body: form });
    state.imageResults = selected.length > 1 ? data.results : [data];
    const select = $("#imageResultSelect");
    select.innerHTML = state.imageResults
      .map((item, index) => `<option value="${index}">${item.filename || `图片 ${index + 1}`}</option>`)
      .join("");
    renderImageResult(0);
    setMessage("图片检测完成");
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    $("#detectImageBtn").disabled = false;
  }
}

async function detectVideo() {
  const input = $("#videoFile");
  if (!input.files.length) {
    setMessage("请选择视频文件", true);
    return;
  }
  await prepareDetectionUpload("video");
  const form = new FormData();
  const p = params();
  form.append("file", input.files[0]);
  form.append("conf", p.conf);
  form.append("iou", p.iou);
  form.append("imgsz", p.imgsz);

  $("#detectVideoBtn").disabled = true;
  $("#videoProgress").value = 0;
  $("#videoTaskStatus").textContent = "上传中";
  $("#resultVideo").removeAttribute("src");
  $("#videoDownload").classList.add("disabled");

  try {
    const task = await apiJson("/api/detect/video", { method: "POST", body: form });
    pollVideoTask(task.status_url);
  } catch (error) {
    $("#detectVideoBtn").disabled = false;
    $("#videoTaskStatus").textContent = "失败";
    setMessage(error.message, true);
  }
}

function pollVideoTask(statusUrl) {
  clearInterval(state.videoPollTimer);
  state.videoPollTimer = setInterval(async () => {
    try {
      const task = await apiJson(statusUrl);
      $("#videoProgress").value = task.progress || 0;
      $("#videoTaskStatus").textContent = `${task.status} ${task.progress || 0}%`;
      renderVideoSummary(task.summary_stats || {});
      if (task.status === "done") {
        clearInterval(state.videoPollTimer);
        $("#detectVideoBtn").disabled = false;
        $("#resultVideo").src = withCacheBust(task.result_video_url);
        $("#videoDownload").href = task.result_video_url;
        $("#videoDownload").classList.remove("disabled");
        setMessage("视频检测完成");
      }
      if (task.status === "failed") {
        clearInterval(state.videoPollTimer);
        $("#detectVideoBtn").disabled = false;
        setMessage(task.message || "视频处理失败", true);
      }
    } catch (error) {
      clearInterval(state.videoPollTimer);
      $("#detectVideoBtn").disabled = false;
      setMessage(error.message, true);
    }
  }, 1200);
}

function renderVideoSummary(summary = {}) {
  const entries = Object.entries(summary);
  $("#videoSummary").innerHTML = entries.length
    ? entries.map(([key, value]) => `<tr><th>${key}</th><td>${value}</td></tr>`).join("")
    : `<tr><td>暂无视频统计</td></tr>`;
}

async function startStream() {
  if (state.socket) {
    await stopStream();
  }
  const p = params();
  $("#streamStatus").textContent = "连接中";
  try {
    const data = await apiJson("/api/stream/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: $("#streamSource").value || "0",
        conf: Number(p.conf),
        iou: Number(p.iou),
        imgsz: Number(p.imgsz),
      }),
    });
    state.streamId = data.stream_id;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    state.socket = new WebSocket(`${protocol}://${window.location.host}${data.websocket_url}`);
    state.socket.onopen = () => {
      $("#streamStatus").textContent = "检测中";
    };
    state.socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "error") {
        $("#streamStatus").textContent = "异常";
        setMessage(payload.message, true);
        return;
      }
      $("#streamFrame").src = `data:image/jpeg;base64,${payload.image}`;
      renderStats($("#streamStats"), payload.stats);
      renderDetections($("#streamDetections"), payload.detections);
    };
    state.socket.onclose = () => {
      $("#streamStatus").textContent = "已停止";
      state.socket = null;
    };
    state.socket.onerror = () => {
      $("#streamStatus").textContent = "异常";
      setMessage("实时连接失败", true);
    };
  } catch (error) {
    $("#streamStatus").textContent = "异常";
    setMessage(error.message, true);
  }
}

async function stopStream() {
  const streamId = state.streamId;
  if (state.socket) {
    state.socket.close();
    state.socket = null;
  }
  if (streamId) {
    try {
      await apiJson(`/api/stream/stop/${streamId}`, { method: "POST" });
    } catch (_) {
      // The WebSocket cleanup may already have released the stream.
    }
  }
  state.streamId = null;
  $("#streamStatus").textContent = "已停止";
}

function bindEvents() {
  $("#confRange").addEventListener("input", (event) => {
    $("#confInput").value = event.target.value;
  });
  $("#confInput").addEventListener("input", (event) => {
    $("#confRange").value = event.target.value;
  });
  $("#detectImageBtn").addEventListener("click", detectImages);
  $("#imageFile").addEventListener("change", resetImageUploadView);
  $("#imageResultSelect").addEventListener("change", (event) => renderImageResult(Number(event.target.value)));
  $("#detectVideoBtn").addEventListener("click", detectVideo);
  $("#videoFile").addEventListener("change", resetVideoUploadView);
  $("#startStreamBtn").addEventListener("click", startStream);
  $("#stopStreamBtn").addEventListener("click", stopStream);
  $("#uploadModelBtn").addEventListener("click", uploadModel);
  $("#saveMappingBtn").addEventListener("click", saveMapping);
  $("#activateModelBtn").addEventListener("click", activateModel);
  $("#deleteModelBtn").addEventListener("click", deleteModel);
  $("#modelSelect").addEventListener("change", (event) => {
    state.selectedModelId = event.target.value;
    renderSelectedModel();
  });
}

bindEvents();
renderStats($("#imageStats"), {});
renderStats($("#streamStats"), {});
renderDetections($("#imageDetections"), []);
renderDetections($("#streamDetections"), []);
renderVideoSummary({});
loadHealth();
loadModels();
