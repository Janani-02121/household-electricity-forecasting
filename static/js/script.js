/* Shared front-end helpers: navigation, loader, upload area and Chart.js wrappers. */
document.addEventListener("DOMContentLoaded", function () {
  var burger = document.getElementById("burger"), menu = document.getElementById("menu");
  if (burger) burger.addEventListener("click", function () { menu.classList.toggle("open"); });

  // loading overlay for forms that run heavy backend work
  var loader = document.getElementById("loader"), loaderText = document.getElementById("loader-text");
  document.querySelectorAll("form[data-loading]").forEach(function (f) {
    f.addEventListener("submit", function () {
      if (loaderText) loaderText.textContent = f.getAttribute("data-loading");
      loader.classList.add("show");
    });
  });

  // upload drop area
  var drop = document.getElementById("drop"), input = document.getElementById("file");
  if (drop && input) {
    var show = function () { document.getElementById("fname").textContent = input.files.length ? input.files[0].name : "No file selected"; };
    input.addEventListener("change", show);
    ["dragenter", "dragover"].forEach(function (e) { drop.addEventListener(e, function (ev) { ev.preventDefault(); drop.classList.add("over"); }); });
    ["dragleave", "drop"].forEach(function (e) { drop.addEventListener(e, function (ev) { ev.preventDefault(); drop.classList.remove("over"); }); });
    drop.addEventListener("drop", function (ev) { input.files = ev.dataTransfer.files; show(); });
  }

  // forecast form: choosing an end date overrides the "days" field
  var end = document.getElementById("end_date"), days = document.getElementById("days");
  if (end && days) end.addEventListener("input", function () { days.disabled = !!end.value; });
  // auto-dismiss success messages
  setTimeout(function () { document.querySelectorAll(".flash.success").forEach(function (f) { f.remove(); }); }, 9000);
});

var ELF = (function () {
  var COLORS = ["#ffb81c", "#2dd4bf", "#ff6b6b", "#a78bfa", "#60a5fa"];
  var registry = {};
  if (window.Chart) {
    Chart.defaults.color = "#93a4c0";
    Chart.defaults.borderColor = "#223454";
    Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
  }
  function make(id, type, labels, datasets, opts) {
    var el = document.getElementById(id);
    if (!el || !window.Chart) return;
    opts = opts || {};
    if (registry[id]) registry[id].destroy();
    registry[id] = new Chart(el, {
      type: type,
      data: { labels: labels, datasets: datasets },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 700 },
        interaction: { mode: "index", intersect: false },
        plugins: { legend: { display: datasets.length > 1 || opts.legend }, title: { display: !!opts.title, text: opts.title } },
        scales: type === "doughnut" ? {} : {
          x: { ticks: { maxTicksLimit: 12, maxRotation: 0 }, grid: { display: false } },
          y: { beginAtZero: opts.zero !== false, title: { display: !!opts.y, text: opts.y } }
        }
      }
    });
  }
  function ds(label, data, i, extra) {
    var c = COLORS[i % COLORS.length];
    return Object.assign({ label: label, data: data, borderColor: c, backgroundColor: c + "33", borderWidth: 2, pointRadius: 0, tension: 0.25 }, extra || {});
  }
  return {
    line: function (id, labels, series, opts) {
      make(id, "line", labels, series.map(function (s, i) { return ds(s.label, s.data, i, s.extra); }), opts);
    },
    area: function (id, labels, label, data, opts) {
      make(id, "line", labels, [ds(label, data, 0, { fill: true })], opts);
    },
    bar: function (id, labels, label, data, opts) {
      var cols = labels.map(function (_, i) { return COLORS[(opts && opts.multi ? i : 0) % COLORS.length] + "cc"; });
      make(id, "bar", labels, [{ label: label, data: data, backgroundColor: cols, borderRadius: 6 }], opts);
    },
    hbar: function (id, labels, label, data, opts) {
      make(id, "bar", labels, [{ label: label, data: data, backgroundColor: "#2dd4bfcc", borderRadius: 6 }], Object.assign({}, opts));
      if (registry[id]) { registry[id].options.indexAxis = "y"; registry[id].update(); }
    }
  };
})();
