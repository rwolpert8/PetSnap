/* Same-origin demo: no API secrets or third-party image uploads in the browser. */
const $ = (id) => document.getElementById(id);
const fileInput = $('file-input');
const dropZone = $('drop-zone');
const predictButton = $('predict-button');
const samples = [...document.querySelectorAll('[data-sample]')];
const allowedTypes = new Set(['image/jpeg', 'image/png', 'image/webp']);
const maxBytes = 10 * 1024 * 1024;
let selectedFile = null;
let previewUrl = null;
let request = null;
let selectionVersion = 0;
let dragDepth = 0;

function announce(message) {
  $('announcement').textContent = message;
}

function showError(message) {
  $('error-message').textContent = message;
  $('error-message').hidden = false;
}

function clearError() {
  $('error-message').hidden = true;
  $('error-message').textContent = '';
}

function setResultState(state) {
  $('result-empty').hidden = state !== 'empty';
  $('result-loading').hidden = state !== 'loading';
  $('result-content').hidden = state !== 'ready';
  $('result-panel').setAttribute('aria-busy', String(state === 'loading'));
  predictButton.disabled = state === 'loading' || !selectedFile;
  $('predict-label').textContent = state === 'loading' ? 'Sniffing out the possibilities…' : 'Find my dog’s breed';
}

function cancelRequest() {
  if (request) request.abort();
  request = null;
}

function resetSelection() {
  selectionVersion += 1;
  cancelRequest();
  selectedFile = null;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  fileInput.value = '';
  $('preview-image').removeAttribute('src');
  $('preview-content').hidden = true;
  $('upload-empty').hidden = false;
  $('selected-file').hidden = true;
  samples.forEach((button) => button.setAttribute('aria-pressed', 'false'));
  clearError();
  setResultState('empty');
}

async function selectFile(file, sampleName = null) {
  if (!file) return;
  resetSelection();
  const version = selectionVersion;
  if (!allowedTypes.has(file.type)) {
    showError('That format is a little unfamiliar. Choose a JPG, PNG, or WebP photo.');
    return;
  }
  if (!file.size || file.size > maxBytes) {
    showError('Choose a photo smaller than 10 MB. A smaller copy of your photo works too.');
    return;
  }
  const url = URL.createObjectURL(file);
  const probe = new Image();
  probe.src = url;
  try {
    await probe.decode();
    if (probe.naturalWidth * probe.naturalHeight > 20_000_000) {
      throw new Error('This photo is a little too big. Resize it to fewer than 20 million pixels.');
    }
  } catch (error) {
    URL.revokeObjectURL(url);
    if (version !== selectionVersion) return;
    showError(error.message.startsWith('This photo') ? error.message : 'We couldn’t open that photo. Try a different JPG, PNG, or WebP image.');
    return;
  }
  if (version !== selectionVersion) {
    URL.revokeObjectURL(url);
    return;
  }
  selectedFile = file;
  previewUrl = url;
  $('preview-image').src = url;
  $('upload-empty').hidden = true;
  $('preview-content').hidden = false;
  $('selected-file').hidden = false;
  $('file-name').textContent = `${file.name} · ${Math.max(1, Math.round(file.size / 1024))} KB`;
  samples.forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.sample === sampleName)));
  setResultState('empty');
  announce('Photo ready. Select Find my dog’s breed to get your matches.');
}

function renderPredictions(data) {
  const predictions = data.top_5_predictions;
  if (!Array.isArray(predictions) || predictions.length !== 5 || predictions.some((item) =>
    typeof item.breed !== 'string' || !Number.isFinite(item.confidence) || item.confidence < 0 || item.confidence > 100)) {
    throw new Error('The model returned an unexpected result. Please try again.');
  }
  const top = predictions[0];
  $('top-breed').textContent = top.breed;
  $('top-score').textContent = `${top.confidence.toFixed(1)}%`;
  $('prediction-list').replaceChildren(...predictions.map((prediction, index) => {
    const row = document.createElement('div');
    row.className = 'prediction-row';
    const rank = document.createElement('span');
    rank.className = 'prediction-rank';
    rank.textContent = String(index + 1).padStart(2, '0');
    const breed = document.createElement('span');
    breed.textContent = prediction.breed;
    const value = document.createElement('span');
    value.className = 'prediction-value';
    value.textContent = `${prediction.confidence.toFixed(1)}%`;
    const bar = document.createElement('div');
    bar.className = 'prediction-bar';
    bar.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('span');
    fill.style.width = `${prediction.confidence}%`;
    bar.append(fill);
    row.append(rank, breed, value, bar);
    return row;
  }));
  $('result-caveat').textContent = 'These scores compare visual resemblance among the model’s 120 breeds. They aren’t ancestry percentages or a guarantee of breed identity.';
  setResultState('ready');
  announce(`Closest match: ${top.breed}, with a model score of ${top.confidence.toFixed(1)} percent. Five matches are ready.`);
  if (window.matchMedia('(max-width: 580px)').matches) {
    $('result-panel').scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' });
  }
}

async function predict() {
  if (!selectedFile || request) return;
  clearError();
  const version = selectionVersion;
  const controller = new AbortController();
  request = controller;
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 60_000);
  const form = new FormData();
  form.append('file', selectedFile);
  setResultState('loading');
  announce('Looking for your dog’s closest breed matches.');
  try {
    const response = await fetch('/api/demo/predict', { method: 'POST', body: form, signal: controller.signal });
    const data = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(typeof data?.detail === 'string' ? data.detail : 'The model is unavailable right now. Please try again in a moment.');
    }
    if (version !== selectionVersion) return;
    if (!data) throw new Error('The model returned an unexpected result. Please try again.');
    renderPredictions(data);
  } catch (error) {
    if (version !== selectionVersion) return;
    setResultState('empty');
    if (timedOut) showError('This is taking longer than expected. Give the model a moment, then try again.');
    else if (error.name !== 'AbortError') {
      showError(error instanceof TypeError ? 'We couldn’t reach PetSnap. Check your connection and try again.' : error.message);
    }
  } finally {
    clearTimeout(timeout);
    if (request === controller) request = null;
  }
}

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => selectFile(fileInput.files[0]));
$('remove-photo').addEventListener('click', () => { resetSelection(); dropZone.focus(); announce('Photo removed.'); });
$('try-another').addEventListener('click', () => { resetSelection(); dropZone.focus(); });
predictButton.addEventListener('click', predict);

dropZone.addEventListener('dragenter', (event) => { event.preventDefault(); dragDepth += 1; dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragover', (event) => { event.preventDefault(); event.dataTransfer.dropEffect = 'copy'; });
dropZone.addEventListener('dragleave', () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', (event) => {
  event.preventDefault();
  dragDepth = 0;
  dropZone.classList.remove('drag-over');
  const files = event.dataTransfer.files;
  if (files.length !== 1) { showError('One good dog at a time. Please choose a single photo.'); return; }
  selectFile(files[0]);
});
// Prevent an image dropped outside the target from navigating away from the app.
window.addEventListener('dragover', (event) => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); });
window.addEventListener('drop', (event) => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); });

samples.forEach((button) => {
  button.setAttribute('aria-pressed', 'false');
  button.addEventListener('click', async () => {
    resetSelection();
    const version = selectionVersion;
    const name = button.dataset.sample;
    announce('Loading sample photo.');
    try {
      const response = await fetch(`/assets/${name}.jpg`);
      if (!response.ok) throw new Error('This sample photo isn’t available. Try uploading your own.');
      const blob = await response.blob();
      if (version !== selectionVersion) return;
      await selectFile(new File([blob], `${name}-sample.jpg`, { type: 'image/jpeg' }), name);
    } catch (error) {
      if (version === selectionVersion) showError('This sample photo couldn’t load. Try uploading your own.');
    }
  });
});
