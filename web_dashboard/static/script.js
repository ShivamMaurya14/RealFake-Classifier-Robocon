// Robocon Real-Time Perception & Decision Visualizer Client
document.addEventListener('DOMContentLoaded', () => {
    const backendSelect = document.getElementById('backend-select');
    const activeBackendBadge = document.getElementById('active-backend-badge');
    const decisionCard = document.getElementById('decision-card');
    const actionText = document.getElementById('action-text');
    const statusMsg = document.getElementById('status-msg');
    const badgeLock = document.getElementById('badge-lock');

    const btnTogglePipeline = document.getElementById('btn-toggle-pipeline');
    const btnPowerIcon = document.getElementById('btn-power-icon');
    const btnPowerText = document.getElementById('btn-power-text');
    const systemStatusPill = document.getElementById('system-status-pill');
    const systemStatusText = document.getElementById('system-status-text');

    const valClass = document.getElementById('val-class');
    const valConf = document.getElementById('val-conf');
    const confProgressBar = document.getElementById('conf-progress-bar');
    const valHeading = document.getElementById('val-heading');
    const valOffsetX = document.getElementById('val-offset-x');
    const valDistance = document.getElementById('val-distance');
    const valGripper = document.getElementById('val-gripper');

    const metricFps = document.getElementById('metric-fps');
    const metricLatency = document.getElementById('metric-latency');
    const rosLogTerminal = document.getElementById('ros-log-terminal');

    // Power / Hardware Stop Button
    if (btnTogglePipeline) {
        btnTogglePipeline.addEventListener('click', () => {
            fetch('/api/toggle_pipeline', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: 'toggle' })
            })
            .then(res => res.json())
            .then(data => {
                updatePowerUI(data.is_active);
            })
            .catch(err => console.error("Error toggling pipeline:", err));
        });
    }

    function updatePowerUI(isActive) {
        if (!btnTogglePipeline) return;
        if (isActive) {
            btnTogglePipeline.className = 'btn-power btn-stop';
            if (btnPowerIcon) btnPowerIcon.innerText = '⏹';
            if (btnPowerText) btnPowerText.innerText = 'STOP PERCEPTION';
            if (systemStatusPill) {
                systemStatusPill.style.borderColor = '#10b981';
                systemStatusPill.style.color = '#10b981';
                systemStatusPill.style.background = 'rgba(16, 185, 129, 0.15)';
            }
            if (systemStatusText) systemStatusText.innerText = 'NODE ACTIVE';
        } else {
            btnTogglePipeline.className = 'btn-power btn-start';
            if (btnPowerIcon) btnPowerIcon.innerText = '▶';
            if (btnPowerText) btnPowerText.innerText = 'START PERCEPTION';
            if (systemStatusPill) {
                systemStatusPill.style.borderColor = '#ef4444';
                systemStatusPill.style.color = '#ef4444';
                systemStatusPill.style.background = 'rgba(239, 68, 68, 0.15)';
            }
            if (systemStatusText) systemStatusText.innerText = 'STANDBY (0% CPU)';
        }
    }

    // Hot-swap backend selector
    backendSelect.addEventListener('change', () => {
        const selected = backendSelect.value;
        fetch('/api/backend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ backend: selected })
        })
        .then(res => res.json())
        .then(data => {
            console.log("Switched backend:", data);
        })
        .catch(err => console.error("Error switching backend:", err));
    });

    // Telemetry Polling Loop (15 Hz)
    function fetchTelemetry() {
        fetch('/api/status')
            .then(res => res.json())
            .then(data => {
                // Update FPS & Latency
                if (metricFps) metricFps.innerText = data.fps.toFixed(1);
                if (metricLatency) metricLatency.innerText = `${data.latency_ms.toFixed(1)} ms`;

                // Update Backend Badge
                if (activeBackendBadge) activeBackendBadge.innerText = data.backend;

                // Update Action & Robot Decision
                if (actionText) actionText.innerText = data.action;
                if (statusMsg) statusMsg.innerText = data.status_message;

                // Update Action Card Styling
                if (decisionCard) {
                    decisionCard.classList.remove('real-target', 'fake-target');
                    if (data.classification === 'REAL') {
                        decisionCard.classList.add('real-target');
                    } else if (data.classification === 'FAKE') {
                        decisionCard.classList.add('fake-target');
                    }
                }

                if (badgeLock) {
                    if (data.target_locked) {
                        badgeLock.innerText = 'TARGET LOCKED';
                        badgeLock.style.background = 'rgba(16, 185, 129, 0.2)';
                        badgeLock.style.color = '#10b981';
                    } else if (data.classification === 'FAKE') {
                        badgeLock.innerText = 'OBSTACLE';
                        badgeLock.style.background = 'rgba(239, 68, 68, 0.2)';
                        badgeLock.style.color = '#ef4444';
                    } else {
                        badgeLock.innerText = 'SEARCHING';
                        badgeLock.style.background = 'rgba(245, 158, 11, 0.15)';
                        badgeLock.style.color = '#f59e0b';
                    }
                }

                // Update Target Classification
                if (valClass) {
                    valClass.innerText = data.classification;
                    valClass.style.color = data.classification === 'REAL' ? '#10b981' : (data.classification === 'FAKE' ? '#ef4444' : '#fff');
                }

                const confPercent = (data.confidence * 100).toFixed(1);
                if (valConf) valConf.innerText = `${confPercent}%`;
                if (confProgressBar) {
                    confProgressBar.style.width = `${confPercent}%`;
                    confProgressBar.style.background = data.classification === 'REAL' ? '#10b981' : (data.classification === 'FAKE' ? '#ef4444' : '#3b82f6');
                }

                // Update Kinematics
                if (valHeading) valHeading.innerText = `${data.heading_error_deg > 0 ? '+' : ''}${data.heading_error_deg.toFixed(1)}°`;
                if (valOffsetX) valOffsetX.innerText = `${data.offset_x} px`;
                if (valDistance) valDistance.innerText = data.distance_proxy.toFixed(2);
                if (valGripper) {
                    valGripper.innerText = data.gripper_command;
                    valGripper.style.color = data.gripper_command === 'CLOSE' ? '#10b981' : (data.gripper_command === 'BYPASS' ? '#ef4444' : '#38bdf8');
                }

                // Update ROS 2 Topic Terminal Log
                if (rosLogTerminal) {
                    rosLogTerminal.innerText = JSON.stringify({
                        topic: "/perception/decision",
                        timestamp: new Date().toISOString().substring(11, 19),
                        action: data.action,
                        target: data.classification,
                        confidence: data.confidence,
                        heading_err: data.heading_error_deg,
                        gripper: data.gripper_command
                    }, null, 2);
                }
            })
            .catch(err => console.error("Telemetry fetch error:", err));
    }

    setInterval(fetchTelemetry, 70);
});
