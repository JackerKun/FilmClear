(() => {
  "use strict";

  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
  const svgNS = "http://www.w3.org/2000/svg";
  const svgElement = (name) => document.createElementNS(svgNS, name);

  function activate(card) {
    const viewer = card.querySelector(".viewer");
    const toolbar = card.querySelector(".toolbar");
    const marks = JSON.parse(card.querySelector("template").content.textContent);
    const frame = [0, 0, Number(card.dataset.width), Number(card.dataset.height)];
    const panes = Array.from(viewer.querySelectorAll(".viewport"), (viewport) => {
      const img = viewport.querySelector("img");
      return { viewport, img, original: img.getAttribute("src"),
        stage: viewport.querySelector(".stage"), svg: viewport.querySelector("svg") };
    });
    const markList = card.querySelector(".defect-list");
    const level = toolbar.querySelector("output");
    const toggle = toolbar.querySelector(".mark-toggle input");
    const split = toolbar.querySelector(".split-control");
    const cropButton = toolbar.querySelector('[data-action="whole"]');
    let region = frame;
    let active = -1;
    let zoom = 1;
    let center = [frame[2] / 2, frame[3] / 2];
    let drag = null;
    let suppressClick = false;

    function fit(pane) {
      return Math.min(pane.viewport.clientWidth / region[2], pane.viewport.clientHeight / region[3]);
    }

    function constrain() {
      const maxSpan = [0, 0];
      for (const pane of panes) {
        const scale = fit(pane) * zoom;
        if (scale <= 0) continue;
        maxSpan[0] = Math.max(maxSpan[0], pane.viewport.clientWidth / (2 * scale));
        maxSpan[1] = Math.max(maxSpan[1], pane.viewport.clientHeight / (2 * scale));
      }
      center = center.map((value, axis) => {
        const half = Math.min(maxSpan[axis], region[axis + 2] / 2);
        return clamp(value, region[axis] + half, region[axis] + region[axis + 2] - half);
      });
    }

    function render() {
      constrain();
      viewer.classList.toggle("is-zoomed", zoom > 1.001);
      level.value = `${zoom.toFixed(1)}×`;
      for (const pane of panes) {
        const scale = fit(pane) * zoom;
        if (scale <= 0) continue;
        const { viewport, stage } = pane;
        stage.style.width = `${region[2] * scale}px`;
        stage.style.height = `${region[3] * scale}px`;
        stage.style.left = `${viewport.clientWidth / 2 - (center[0] - region[0]) * scale}px`;
        stage.style.top = `${viewport.clientHeight / 2 - (center[1] - region[1]) * scale}px`;
        for (const ring of pane.rings) {
          ring.circle.setAttribute("r", String(8 / scale));
          ring.circle.style.display = ring.span * scale < 17 ? "" : "none";
        }
      }
    }

    function reset() {
      zoom = 1;
      center = [region[0] + region[2] / 2, region[1] + region[3] / 2];
      render();
    }

    function setZoom(value, pane, pointer) {
      const old = zoom;
      zoom = clamp(value, 1, 20);
      if (pane && pointer) {
        const rectangle = pane.viewport.getBoundingClientRect();
        const base = fit(pane);
        if (base > 0) {
          const anchor = [
            pointer.clientX - rectangle.left - pane.viewport.clientWidth / 2,
            pointer.clientY - rectangle.top - pane.viewport.clientHeight / 2,
          ];
          center = center.map((position, axis) => position + anchor[axis] * (1 / old - 1 / zoom) / base);
        }
      }
      render();
    }

    function setRegion(index) {
      active = index;
      region = index < 0 ? frame : marks[index].crop;
      cropButton.disabled = index < 0;
      card.querySelector(".scope").textContent = index < 0 ? "整张预览" : marks[index].label;
      for (const [paneIndex, pane] of panes.entries()) {
        pane.img.src = index < 0 ? pane.original : marks[index].images[paneIndex];
        pane.svg.setAttribute("viewBox", region.join(" "));
        for (const [markIndex, marker] of pane.markers.entries()) {
          marker.classList.toggle("is-active", markIndex === index);
        }
      }
      for (const [markIndex, button] of Array.from(markList.children).entries()) {
        button.setAttribute("aria-pressed", String(markIndex === index));
      }
      reset();
    }

    for (const [index, mark] of marks.entries()) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = mark.status;
      button.textContent = mark.label;
      button.addEventListener("click", () => setRegion(index));
      markList.append(button);
    }

    for (const pane of panes) {
      const { viewport, svg } = pane;
      pane.rings = [];
      pane.markers = marks.map((mark, index) => {
        const group = svgElement("g");
        group.classList.add("marker", mark.status);
        const path = svgElement("path");
        path.setAttribute("d", mark.path);
        const [x, y, width, height] = mark.box;
        const circle = svgElement("circle");
        circle.setAttribute("cx", String(x + width / 2));
        circle.setAttribute("cy", String(y + height / 2));
        pane.rings.push({ circle, span: Math.max(width, height) });
        const title = svgElement("title");
        title.textContent = mark.label;
        group.append(title, path, circle);
        group.addEventListener("click", (event) => {
          if (!suppressClick) setRegion(index);
          event.stopPropagation();
        });
        svg.append(group);
        return group;
      });

      viewport.addEventListener("wheel", (event) => {
        if (!event.ctrlKey && !event.metaKey) return;
        event.preventDefault();
        setZoom(zoom * Math.exp(-clamp(event.deltaY, -200, 200) * 0.003), pane, event);
      }, { passive: false });

      viewport.addEventListener("pointerdown", (event) => {
        suppressClick = false;
        if (zoom <= 1.001 || event.button !== 0) return;
        drag = { pointer: event.pointerId, pane,
          x: event.clientX, y: event.clientY, moved: false };
      });
      viewport.addEventListener("pointermove", (event) => {
        if (!drag || drag.pointer !== event.pointerId || drag.pane !== pane) return;
        const dx = event.clientX - drag.x;
        const dy = event.clientY - drag.y;
        if (!drag.moved && Math.hypot(dx, dy) < 4) return;
        if (!drag.moved) {
          viewport.setPointerCapture(event.pointerId);
          viewer.classList.add("is-dragging");
          drag.moved = true;
        }
        const scale = fit(pane) * zoom;
        center[0] -= dx / scale;
        center[1] -= dy / scale;
        drag.x = event.clientX;
        drag.y = event.clientY;
        render();
      });
      for (const name of ["pointerup", "pointercancel", "lostpointercapture", "pointerleave"]) {
        viewport.addEventListener(name, (event) => {
          if (!drag || drag.pointer !== event.pointerId || drag.pane !== pane) return;
          if (name === "pointerleave" && drag.moved) return;
          suppressClick = drag.moved;
          drag = null;
          viewer.classList.remove("is-dragging");
        });
      }

      viewport.addEventListener("keydown", (event) => {
        if (event.key === "+" || event.key === "=") setZoom(zoom * 1.4);
        else if (event.key === "-") setZoom(zoom / 1.4);
        else if (event.key === "0") reset();
        else return;
        event.preventDefault();
      });
    }

    const actions = {
      in: () => setZoom(zoom * 1.4),
      out: () => setZoom(zoom / 1.4),
      fit: reset,
      whole: () => setRegion(-1),
      previous: () => setRegion(active < 0 ? marks.length - 1 : (active - 1 + marks.length) % marks.length),
      next: () => setRegion((active + 1) % marks.length),
    };
    for (const button of toolbar.querySelectorAll("[data-action]")) {
      button.addEventListener("click", actions[button.dataset.action]);
    }
    for (const button of toolbar.querySelectorAll("[data-layout]")) {
      button.addEventListener("click", () => {
        const swipe = button.dataset.layout === "swipe";
        viewer.classList.toggle("swipe", swipe);
        viewer.classList.toggle("stack", button.dataset.layout === "stack");
        split.hidden = !swipe;
        for (const choice of toolbar.querySelectorAll("[data-layout]")) {
          choice.setAttribute("aria-pressed", String(choice === button));
        }
        render();
      });
    }
    const slider = split.querySelector("input");
    const divider = viewer.querySelector(".divider");
    function setSplit(value) {
      slider.value = clamp(Math.round(value), 0, 100);
      viewer.style.setProperty("--split", `${slider.value}%`);
    }
    function dragDivider(event) {
      const { left, width } = viewer.getBoundingClientRect();
      setSplit(100 * (event.clientX - left) / width);
    }
    slider.addEventListener("input", () => setSplit(Number(slider.value)));
    divider.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      divider.setPointerCapture(event.pointerId);
      dragDivider(event);
    });
    divider.addEventListener("pointermove", (event) => {
      if (divider.hasPointerCapture(event.pointerId)) dragDivider(event);
    });
    toggle.addEventListener("change", () => viewer.classList.toggle("hide-marks", !toggle.checked));
    toggle.disabled = marks.length === 0;
    toolbar.querySelector('[data-action="previous"]').disabled = marks.length === 0;
    toolbar.querySelector('[data-action="next"]').disabled = marks.length === 0;
    viewer.classList.add("interactive");
    toolbar.hidden = false;
    new ResizeObserver(render).observe(viewer);
    setRegion(-1);
  }

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      observer.unobserve(entry.target);
      activate(entry.target);
    }
  }, { rootMargin: "280px" });
  document.querySelectorAll("article[data-width]").forEach((card) => observer.observe(card));
})();
