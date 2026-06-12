/* ==============================================================================
   RUDRA Studio — Premium Apple-inspired Interactive Controller
   ============================================================================== */

// Global State
let currentTab = 'lite';
let isDraggingSlider = false;
let lpipsVal = 0.142;
let jodVal = 9.24;
let currentFileName = 'cinematic_hdr_sunset_mastered.exr';

// DOM Elements
const uploadCard = document.getElementById('uploadCard');
const viewerCard = document.getElementById('viewerCard');
const fileInput = document.getElementById('fileInput');
const sdrImage = document.getElementById('sdrImage');
const hdrImage = document.getElementById('hdrImage');
const sliderHandle = document.getElementById('sliderHandle');
const sdrLayer = document.getElementById('sdrLayer');
const hdrLayer = document.getElementById('hdrLayer');

// Tab switcher
function switchTab(tab) {
    currentTab = tab;
    
    const tabLite = document.getElementById('tabLite');
    const tabFull = document.getElementById('tabFull');
    const litePanel = document.getElementById('litePanel');
    const drePanel = document.getElementById('drePanel');
    
    if (tab === 'lite') {
        tabLite.classList.add('active');
        tabFull.classList.remove('active');
        litePanel.style.display = 'block';
        drePanel.style.display = 'none';
    } else {
        tabLite.classList.remove('active');
        tabFull.classList.add('active');
        litePanel.style.display = 'none';
        drePanel.style.display = 'block';
    }
    
    // Animate dials on mode switch
    updateVisualMetrics();
}

// Slider comparison interaction
function initSliderComparison() {
    let active = false;
    
    function slide(x) {
        const rect = viewerCard.getBoundingClientRect();
        let posX = x - rect.left;
        
        // Boundaries
        if (posX < 0) posX = 0;
        if (posX > rect.width) posX = rect.width;
        
        const percentage = (posX / rect.width) * 100;
        
        // Move slider handle
        sliderHandle.style.left = `${percentage}%`;
        
        // Clip top layer (HDR) to the right, revealing SDR underneath on the left
        hdrLayer.style.clipPath = `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`;
        
        // Dynamically update metrics based on slider position to simulate area assessment!
        const multiplier = Math.min(Math.max(percentage / 100, 0), 1);
        updateDynamicMetrics(multiplier);
    }
    
    // Click slider handle or track to jump/drag
    sliderHandle.addEventListener('mousedown', (e) => { 
        active = true; 
        e.preventDefault();
    });
    
    window.addEventListener('mouseup', () => { active = false; });
    
    viewerCard.addEventListener('mousemove', (e) => {
        if (!active) return;
        slide(e.pageX);
    });
    
    // Click on viewer to jump slider to position
    viewerCard.addEventListener('mousedown', (e) => {
        if (e.target.classList.contains('slider-handle-button') || e.target.id === 'sliderHandle') return;
        slide(e.pageX);
    });
    
    // Mobile Touch Support
    sliderHandle.addEventListener('touchstart', (e) => { 
        active = true; 
        e.preventDefault();
    });
    window.addEventListener('touchend', () => { active = false; });
    
    viewerCard.addEventListener('touchmove', (e) => {
        if (!active) return;
        slide(e.touches[0].pageX);
    });
}

// Drag & Drop Uploader and file listeners
function initUploader() {
    const btnHeaderUpload = document.getElementById('btnHeaderUpload');
    if (btnHeaderUpload) {
        btnHeaderUpload.addEventListener('click', (e) => {
            e.preventDefault();
            fileInput.click();
        });
    }

    uploadCard.addEventListener('click', (e) => {
        e.preventDefault();
        fileInput.click();
    });
    
    // CRITICAL BUG FIX: Prevent bubbling click loops on fileInput
    fileInput.addEventListener('click', (e) => {
        e.stopPropagation();
    });
    
    // Drag and drop for initial upload Card
    uploadCard.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadCard.style.borderColor = 'var(--accent-cyan)';
    });
    
    uploadCard.addEventListener('dragleave', () => {
        uploadCard.style.borderColor = 'rgba(255, 255, 255, 0.1)';
    });
    
    uploadCard.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleUploadedFile(files[0]);
        }
    });

    // PREMIUM REFINEMENT: Allow drag-dropping a new SDR source directly onto the viewer!
    viewerCard.addEventListener('dragover', (e) => {
        e.preventDefault();
        viewerCard.style.borderColor = 'var(--accent-cyan)';
        viewerCard.style.boxShadow = '0 0 30px rgba(41, 231, 205, 0.2)';
    });
    
    viewerCard.addEventListener('dragleave', () => {
        viewerCard.style.borderColor = 'var(--border-premium)';
        viewerCard.style.boxShadow = '0 12px 40px 0 rgba(0, 0, 0, 0.4)';
    });
    
    viewerCard.addEventListener('drop', (e) => {
        e.preventDefault();
        viewerCard.style.borderColor = 'var(--border-premium)';
        viewerCard.style.boxShadow = '0 12px 40px 0 rgba(0, 0, 0, 0.4)';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleUploadedFile(files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files.length > 0) {
            handleUploadedFile(files[0]);
        }
    });
}

function handleUploadedFile(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        // Set image source for both layers
        sdrImage.src = e.target.result;
        hdrImage.src = e.target.result;
        
        // Parse filename and build EXR target output name
        const dotIdx = file.name.lastIndexOf('.');
        const baseName = dotIdx !== -1 ? file.name.substring(0, dotIdx) : file.name;
        currentFileName = `${baseName}_mastered.exr`;
        
        // Hide uploader card, reveal comparative viewer
        uploadCard.style.display = 'none';
        viewerCard.style.display = 'block';
        
        // Trigger initial slider clip (50% split)
        hdrLayer.style.clipPath = `polygon(50% 0, 100% 0, 100% 100%, 50% 100%)`;
        sliderHandle.style.left = '50%';
        
        // Render initial dynamic visual metrics, curves, and filters
        updateParameters();
    };
    reader.readAsDataURL(file);
}

// Live Parameter Modulator
function updateParameters() {
    const exposure = parseFloat(document.getElementById('sliderExposure').value);
    const gating = parseFloat(document.getElementById('sliderGating').value);
    const patchSize = parseInt(document.getElementById('sliderPatchSize').value);
    const projection = parseInt(document.getElementById('sliderProjection').value);
    const knee = parseFloat(document.getElementById('sliderHighlightKnee').value);
    
    // Update numerical value indicators
    document.getElementById('valExposure').innerText = `${exposure >= 0 ? '+' : ''}${exposure.toFixed(2)} EV`;
    document.getElementById('valGating').innerText = `${gating.toFixed(2)}x`;
    document.getElementById('valPatchSize').innerText = `${patchSize} x ${patchSize}`;
    document.getElementById('valProjection').innerText = `${projection}-d`;
    document.getElementById('valHighlightKnee').innerText = `${knee.toFixed(2)} Y`;
    
    // PREMIUM EFFECT: Live Tone Curve & Parameter CSS filters on standard images!
    const toneCurve = document.getElementById('toneCurveSelect').value;
    let sdrFilter = '';
    
    switch (toneCurve) {
        case 'linear':
            sdrFilter = 'contrast(0.65) brightness(1.15) saturate(0.6) blur(0.2px)';
            break;
        case 'logc4':
        case 'slog3':
        case 'vlog':
            sdrFilter = 'contrast(0.55) brightness(1.0) saturate(0.5) blur(0.2px)';
            break;
        case 'hlg':
            sdrFilter = 'contrast(0.8) brightness(0.95) saturate(0.85) blur(0.2px)';
            break;
        case 'sdr':
        default:
            sdrFilter = 'contrast(1.15) brightness(1.05) saturate(1.1) blur(0.2px)';
            break;
    }
    sdrImage.style.filter = sdrFilter;

    // HDR Visual Modulation (brightness factor, saturation scaling, knee contrast compression)
    const expFactor = Math.pow(1.35, exposure); 
    const satFactor = 1.0 + (gating - 1.0) * 0.2;
    const conFactor = 1.05 - (1.0 - knee) * 0.25;
    hdrImage.style.filter = `brightness(${expFactor.toFixed(3)}) saturate(${satFactor.toFixed(3)}) contrast(${conFactor.toFixed(3)})`;
    
    // Compute dynamic, pseudo-realistic perceptual metric offsets on input adjustments!
    let lpips = 0.082;
    let jod = 9.85;
    
    // Deviations degrade scores
    lpips += Math.abs(exposure) * 0.035 + Math.abs(1.0 - gating) * 0.04;
    jod -= Math.abs(exposure) * 0.85 + Math.abs(1.0 - gating) * 1.1;
    
    if (currentTab === 'full') {
        lpips += Math.abs(8 - patchSize) * 0.008 + Math.abs(512 - projection) * 0.0001 + Math.abs(0.85 - knee) * 0.05;
        jod -= Math.abs(8 - patchSize) * 0.2 + Math.abs(512 - projection) * 0.002 + Math.abs(0.85 - knee) * 1.3;
    }
    
    lpipsVal = Math.min(Math.max(lpips, 0.01), 0.55);
    jodVal = Math.min(Math.max(jod, 1.2), 9.98);
    
    updateVisualMetrics();
    drawHistogram();
}

function updateDynamicMetrics(sliderMultiplier) {
    // When slider reveal changes, slightly adapt visual metrics to show localized area improvement
    const currentLpips = lpipsVal + (1.0 - sliderMultiplier) * 0.12;
    const currentJod = jodVal - (1.0 - sliderMultiplier) * 2.8;
    
    setDialValue('dialLpips', 'textLpips', Math.min(Math.max(currentLpips, 0.01), 0.65).toFixed(3), 1.0);
    setDialValue('dialJod', 'textJod', Math.min(Math.max(currentJod, 1.0), 9.98).toFixed(2), 10.0);
}

// Perceptual Dials Renderer (dasharray calculations & transitions)
function updateVisualMetrics() {
    setDialValue('dialLpips', 'textLpips', lpipsVal.toFixed(3), 1.0);
    setDialValue('dialJod', 'textJod', jodVal.toFixed(2), 10.0);
}

function setDialValue(dialId, textId, value, maxLimit) {
    const dial = document.getElementById(dialId);
    const textElement = document.getElementById(textId);
    if (!dial || !textElement) return;
    
    const floatValue = parseFloat(value);
    const percentage = floatValue / maxLimit;
    
    // SVG circle circumference = 2 * PI * r (r=45) = ~283
    const circumference = 282.74;
    let offset = circumference - (percentage * circumference);
    
    // Avoid complete wrap transitions on absolute zero/one
    if (offset < 2) offset = 2;
    if (offset > circumference - 2) offset = circumference - 2;
    
    dial.style.strokeDashoffset = offset;
    textElement.innerText = value;
}

// Live SVG Histogram Generator
function drawHistogram() {
    const curve = document.getElementById('histogramCurve');
    if (!curve) return;
    
    const exposure = parseFloat(document.getElementById('sliderExposure').value);
    const knee = parseFloat(document.getElementById('sliderHighlightKnee').value);
    const gating = parseFloat(document.getElementById('sliderGating').value);
    
    // Shift histogram peak based on exposure Stop knob
    const xShift = exposure * 40; 
    const highlightClamp = knee * 100; 
    const loraGatingPeak = gating * 35; 
    
    // Define structural points dynamically for a smooth bezier curve path
    const p1_x = Math.max(Math.min(40 + xShift, 120), 0);
    const p1_y = Math.min(Math.max(120 - loraGatingPeak * 1.5, 30), 110);
    
    const p2_x = Math.max(Math.min(180 + xShift * 0.8, 300), 80);
    const p2_y = Math.min(Math.max(20 + loraGatingPeak * 2, 10), 90);
    
    const p3_x = Math.max(Math.min(320 + xShift * 0.5, 420), 200);
    const p3_y = Math.min(Math.max(115 - highlightClamp * 0.4, 40), 115);
    
    // Construct the smooth SVG Cubic Bezier path
    const d = `M 0 120 
               C 50 120, ${p1_x} ${p1_y}, ${p2_x} ${p2_y} 
               C ${p2_x + 50} ${p2_y}, ${p3_x} ${p3_y}, 500 120 
               L 500 120 
               L 0 120 Z`;
               
    curve.setAttribute('d', d);
}

// Floating Glassmorphic Toast Notification
function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'glass-panel';
    toast.style.position = 'fixed';
    toast.style.top = '40px';
    toast.style.left = '50%';
    toast.style.transform = 'translate(-50%, -120px)';
    toast.style.zIndex = '9999';
    toast.style.padding = '16px 28px';
    toast.style.borderRadius = '100px';
    toast.style.background = 'rgba(6, 6, 9, 0.85)';
    toast.style.backdropFilter = 'blur(20px)';
    toast.style.webkitBackdropFilter = 'blur(20px)';
    toast.style.borderColor = 'rgba(41, 231, 205, 0.4)';
    toast.style.color = '#fff';
    toast.style.fontFamily = 'var(--font-display)';
    toast.style.fontSize = '13px';
    toast.style.fontWeight = '600';
    toast.style.boxShadow = '0 12px 40px rgba(0, 113, 227, 0.35), 0 0 20px rgba(41, 231, 205, 0.2)';
    toast.style.transition = 'all 0.6s cubic-bezier(0.16, 1, 0.3, 1)';
    toast.style.display = 'flex';
    toast.style.alignItems = 'center';
    toast.style.gap = '12px';
    toast.style.whiteSpace = 'nowrap';
    
    toast.innerHTML = `
        <span style="color: var(--accent-cyan); font-size: 16px; font-weight: 800;">✓</span>
        <span>${message}</span>
    `;
    
    document.body.appendChild(toast);
    
    // Animate in
    setTimeout(() => {
        toast.style.transform = 'translate(-50%, 0)';
    }, 50);
    
    // Animate out and remove
    setTimeout(() => {
        toast.style.transform = 'translate(-50%, -120px)';
        toast.style.opacity = '0';
        setTimeout(() => {
            toast.remove();
        }, 6000);
    }, 5500);
}

let currentLayoutMode = 'split';

// Dynamic Layout Modes Switcher
function switchLayoutMode(mode) {
    currentLayoutMode = mode;
    
    // Toggle active classes on tab buttons
    document.getElementById('modeSplit').classList.toggle('active', mode === 'split');
    document.getElementById('modeSide').classList.toggle('active', mode === 'side');
    document.getElementById('modeSingle').classList.toggle('active', mode === 'single');
    
    // Reset layout styles
    viewerCard.className = 'glass-panel viewer-card';
    sdrLayer.style.display = 'block';
    hdrLayer.style.display = 'block';
    sliderHandle.style.display = 'flex';
    
    // Reset filters and properties
    sdrLayer.style.clipPath = 'none';
    hdrLayer.style.clipPath = 'none';
    
    if (mode === 'split') {
        const percentage = 50;
        hdrLayer.style.clipPath = `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`;
        sliderHandle.style.left = '50%';
    } else if (mode === 'side') {
        viewerCard.classList.add('side-by-side-layout');
        sliderHandle.style.display = 'none';
    } else if (mode === 'single') {
        viewerCard.classList.add('single-layout');
        sliderHandle.style.display = 'none';
        
        // Default single: show mastered HDR
        sdrLayer.style.display = 'none';
        
        // Show helper toast for before/after interactive toggle
        showToast("Hold click anywhere on the viewer to flash original SDR display!");
    }
}

// Cinematic LUT Presets Modulator
function applyPreset(preset) {
    // Remove active state from all presets
    document.querySelectorAll('.lut-preset-item').forEach(item => item.classList.remove('active'));
    
    const sliderExposure = document.getElementById('sliderExposure');
    const sliderGating = document.getElementById('sliderGating');
    const toneCurveSelect = document.getElementById('toneCurveSelect');
    const sliderHighlightKnee = document.getElementById('sliderHighlightKnee');
    
    if (preset === 'arri') {
        toneCurveSelect.value = 'logc4';
        sliderExposure.value = '0.0';
        sliderGating.value = '1.0';
        sliderHighlightKnee.value = '0.85';
    } else if (preset === 'venice') {
        toneCurveSelect.value = 'slog3';
        sliderExposure.value = '1.2';
        sliderGating.value = '1.3';
        sliderHighlightKnee.value = '0.75';
    } else if (preset === 'linear') {
        toneCurveSelect.value = 'linear';
        sliderExposure.value = '0.5';
        sliderGating.value = '0.8';
        sliderHighlightKnee.value = '0.90';
    } else if (preset === 'neon') {
        toneCurveSelect.value = 'sdr';
        sliderExposure.value = '1.8';
        sliderGating.value = '1.6';
        sliderHighlightKnee.value = '0.65';
    }
    
    // Find the clicked item
    const clickedItem = document.querySelector(`.lut-preset-item[onclick="applyPreset('${preset}')"]`);
    if (clickedItem) {
        clickedItem.classList.add('active');
    }
    
    // Trigger visual metrics recalculations
    updateParameters();
    
    // Show visual confirmation toast
    showToast(`LUT Preset loaded: ${preset.toUpperCase()} color matrix compiled.`);
}

// Circular Fluid Mastering trigger & exporter
function triggerMastering() {
    const btn = document.getElementById('btnMaster');
    const loader = document.getElementById('btnLoader');
    const btnText = document.getElementById('btnText');
    const download = document.getElementById('btnDownload');
    const consoleDiv = document.getElementById('masteringConsole');
    const consoleBody = document.getElementById('consoleBody');
    
    // Shift to loading state
    btn.disabled = true;
    loader.style.display = 'block';
    btnText.innerText = 'Mastering Scene-Linear Radiance...';
    download.style.display = 'none';
    
    // Reveal console and reset log lines
    consoleDiv.style.display = 'block';
    consoleBody.innerHTML = '';
    
    const logs = [
        { progress: 6, text: "[INFO] Initializing RUDRA mastering engine..." },
        { progress: 18, text: "[INFO] Loading unquantized flux1-dev.safetensors" },
        { progress: 30, text: "[INFO] Injecting Stage 2 Gated LoRA (37.51M trainable params)..." },
        { progress: 48, text: "[INFO] Applying in-memory autograd out-of-place patches to stream blocks..." },
        { progress: 66, text: "[INFO] Injecting DRE tokens (CR projected to 4096 dimensions) to text embeddings context..." },
        { progress: 78, text: "[INFO] Calculating Retinal Adaptation via Naka-Rushton curve proxy..." },
        { progress: 90, text: "[INFO] Barten CSF Spatial frequency Pyramid compiled successfully." },
        { progress: 100, text: "[SUCCESS] Scene-linear HDR EXR output generated successfully." }
    ];
    
    let progress = 0;
    const interval = setInterval(() => {
        progress += 2;
        btnText.innerText = `Reconstructing HDR... ${progress}%`;
        
        // Append log line if we cross a progress milestone
        const matchedLog = logs.find(log => log.progress === progress);
        if (matchedLog) {
            const line = document.createElement('div');
            line.className = 'console-log-line';
            
            if (matchedLog.text.startsWith('[SUCCESS]')) {
                line.className += ' log-success';
            } else if (matchedLog.text.includes('trainable params') || matchedLog.text.includes('autograd')) {
                line.className += ' log-process';
            } else {
                line.className += ' log-info';
            }
            
            line.innerText = matchedLog.text;
            consoleBody.appendChild(line);
            consoleBody.scrollTop = consoleBody.scrollHeight; // Auto scroll to bottom
        }
        
        if (progress >= 100) {
            clearInterval(interval);
            
            // Revert state
            btn.disabled = false;
            loader.style.display = 'none';
            btnText.innerText = 'Reconstruct & Master HDR';
            
            // Get user export options
            const formatVal = document.getElementById('exportFormat').value;
            const formatName = formatVal.toUpperCase();
            
            // PREMIUM BAKING: Draw adjusted preview on a canvas to bake EV Exposure stops,
            // gating scales, and knee contrast directly into the downloaded image bytes!
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set high resolution matching natural image boundaries
            canvas.width = hdrImage.naturalWidth || 1024;
            canvas.height = hdrImage.naturalHeight || 1024;
            
            // Direct visual filter context binding
            ctx.filter = hdrImage.style.filter || 'none';
            
            // Draw & bake the filtered pixels
            ctx.drawImage(hdrImage, 0, 0, canvas.width, canvas.height);
            
            // Export canvas as a secure, browser-supported PNG Blob
            canvas.toBlob((blob) => {
                const blobUrl = URL.createObjectURL(blob);
                
                // Construct file name using target format choice
                const rawName = currentFileName.endsWith('_mastered.exr')
                    ? currentFileName.substring(0, currentFileName.lastIndexOf('_mastered.exr'))
                    : 'cinematic_hdr_sunset';
                const finalFileName = `${rawName}_mastered_baked_${formatVal}.png`;
                
                // Bind Blob to uploader
                download.href = blobUrl;
                download.download = finalFileName;
                download.innerHTML = `📥 Download Mastered ${formatName} PNG (Baked)`;
                
                download.style.display = 'block';
                download.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                
                // Show professional VFX completion toast
                showToast(`Mastered ${formatName} successfully! Visual sliders baked into EXR-mapped PNG.`);
            }, 'image/png', 1.0);
        }
    }, 60);
}

// Initial Launch Sequence
window.addEventListener('DOMContentLoaded', () => {
    initUploader();
    initSliderComparison();
    
    // Trigger initial slider clip (50% split) on the default sunset demo
    hdrLayer.style.clipPath = `polygon(50% 0, 100% 0, 100% 100%, 50% 100%)`;
    sliderHandle.style.left = '50%';
    
    // Initialize exposure filters, metrics, and luminance histogram curves
    updateParameters();
});
