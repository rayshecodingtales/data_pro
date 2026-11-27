(function () {
  const STORAGE_KEY = "gradscope_2025_records_v1";
  const BASE_DATA = Array.isArray(window.MBA_DATA) ? window.MBA_DATA : [];
  let DATA = [];
  let serverOnline = false;
  let schoolSort = "roi";
  let visibleCount = 12;
  let selectedIds = new Set();

  const uniq = (arr) => [...new Set(arr.filter((v) => v !== null && v !== undefined && v !== ""))].sort();
  const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0);
  const esc = (s) =>
    String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

  // Live Pearson correlation
  function pearson(xs, ys) {
    if (xs.length < 2) return 0;
    const mx = avg(xs), my = avg(ys);
    let num = 0, dx = 0, dy = 0;
    for (let i = 0; i < xs.length; i++) {
      const a = xs[i] - mx, b = ys[i] - my;
      num += a * b;
      dx += a * a;
      dy += b * b;
    }
    return dx && dy ? num / Math.sqrt(dx * dy) : 0;
  }

  // Live 3x3 OLS solver: PackageLPA = b0 + b1*MBACGPA + b2*WorkExpMonths
  function fitOLS(placedRows) {
    if (placedRows.length < 5) {
      return { intercept: 9.7, cgpaCoef: 1.55, expCoef: 0.1, r2: 0, rmse: 2.0 };
    }
    let s00 = placedRows.length, s01 = 0, s02 = 0, s11 = 0, s12 = 0, s22 = 0;
    let t0 = 0, t1 = 0, t2 = 0;
    for (const d of placedRows) {
      const x1 = d.MBACGPA, x2 = d.WorkExpMonths, y = d.PackageLPA;
      s01 += x1; s02 += x2;
      s11 += x1 * x1; s12 += x1 * x2; s22 += x2 * x2;
      t0 += y; t1 += x1 * y; t2 += x2 * y;
    }
    const A = [
      [s00, s01, s02, t0],
      [s01, s11, s12, t1],
      [s02, s12, s22, t2],
    ];
    for (let i = 0; i < 3; i++) {
      let maxR = i;
      for (let r = i + 1; r < 3; r++) {
        if (Math.abs(A[r][i]) > Math.abs(A[maxR][i])) maxR = r;
      }
      [A[i], A[maxR]] = [A[maxR], A[i]];
      const piv = A[i][i] || 1e-9;
      for (let j = i; j < 4; j++) A[i][j] /= piv;
      for (let r = 0; r < 3; r++) {
        if (r === i) continue;
        const f = A[r][i];
        for (let j = i; j < 4; j++) A[r][j] -= f * A[i][j];
      }
    }
    const [b0, b1, b2] = [A[0][3], A[1][3], A[2][3]];
    const my = t0 / placedRows.length;
    let ssRes = 0, ssTot = 0;
    for (const d of placedRows) {
      const pred = b0 + b1 * d.MBACGPA + b2 * d.WorkExpMonths;
      ssRes += (d.PackageLPA - pred) ** 2;
      ssTot += (d.PackageLPA - my) ** 2;
    }
    const r2 = ssTot ? 1 - ssRes / ssTot : 0;
    const rmse = Math.sqrt(ssRes / placedRows.length);
    return { intercept: b0, cgpaCoef: b1, expCoef: b2, r2, rmse };
  }

  function enrichRow(r) {
    const workExp = Number(r.WorkExpMonths) || 0;
    const cgpa = Number(r.MBACGPA) || 0;
    const stipend = Number(r.InternshipStipendPM) || 0;
    const feeLakhs = r.CourseFeeLakhs ? Number(r.CourseFeeLakhs) : Number(r.CourseFeeINR || 1500000) / 100000;
    const isPlaced = r.EmploymentStatus === "Placed" ? 1 : 0;
    const pkg = isPlaced && r.PackageLPA !== null && r.PackageLPA !== "" ? Number(r.PackageLPA) : null;
    return {
      ...r,
      Age: Number(r.Age),
      UndergradPercentage: Number(r.UndergradPercentage),
      Class10Score: Number(r.Class10Score),
      Class12Score: Number(r.Class12Score),
      CATPercentile: Number(r.CATPercentile),
      MBACGPA: cgpa,
      WorkExpMonths: workExp,
      InternshipStipendPM: stipend,
      CourseFeeLakhs: Number(feeLakhs.toFixed(2)),
      IsPlaced: isPlaced,
      PackageLPA: pkg,
      ROIMultiple: isPlaced && pkg ? Number((pkg / feeLakhs).toFixed(2)) : null,
      CGPABand: cgpa < 6.5 ? "5.5-6.5" : cgpa < 7.5 ? "6.5-7.5" : cgpa < 8.5 ? "7.5-8.5" : "8.5-9.8",
      WorkExpBand: workExp === 0 ? "0m (Fresher)" : workExp <= 12 ? "1-12m" : workExp <= 24 ? "13-24m" : workExp <= 36 ? "25-36m" : "37-48m",
      StipendBand: stipend < 15000 ? "<15k" : stipend < 25000 ? "15k-25k" : stipend < 50000 ? "25k-50k" : "50k+",
    };
  }

  function initDataFromStorageOrBase() {
    if (!serverOnline) {
      try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved) {
          DATA = JSON.parse(saved).map(enrichRow);
          return;
        }
      } catch (_) {}
    }
    DATA = BASE_DATA.map(enrichRow);
  }

  function persistLocal() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(DATA));
    } catch (_) {}
  }

  function populateDropdown(id, vals, includeAllLabel) {
    const sel = document.getElementById(id);
    const prev = sel.value;
    sel.innerHTML = includeAllLabel ? `<option value="ALL">${includeAllLabel}</option>` : "";
    vals.forEach((v) => {
      const o = document.createElement("option");
      o.value = v;
      o.textContent = v;
      sel.appendChild(o);
    });
    if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
  }

  function syncDropdownOptions() {
    const colleges = uniq(DATA.map((d) => d.CollegeName));
    const specs = uniq(DATA.map((d) => d.MBASpecialization));
    const states = uniq(DATA.map((d) => d.HomeState));
    const recruiters = uniq(DATA.filter((d) => d.IsPlaced === 1).map((d) => d.RecruiterCompany));
    const locs = uniq(DATA.filter((d) => d.IsPlaced === 1).map((d) => d.JoiningLocation));

    populateDropdown("cSchool", colleges);
    populateDropdown("cSpec", specs);
    populateDropdown("dirSchool", colleges, "All Colleges");
    populateDropdown("dirSpec", specs, "All Specializations");

    populateDropdown("aCollege", colleges);
    populateDropdown("aSpec", specs);
    populateDropdown("aState", states);
    populateDropdown("aRecruiter", recruiters);
    populateDropdown("aLoc", locs);
  }

  function nextStudentId() {
    let maxNum = 4000;
    DATA.forEach((d) => {
      const m = String(d.StudentID).match(/(\d+)$/);
      if (m) maxNum = Math.max(maxNum, parseInt(m[1], 10));
    });
    return `MBA2025-${maxNum + 1}`;
  }

  function renderAll() {
    syncDropdownOptions();
    const placed = DATA.filter((d) => d.IsPlaced === 1);
    const ols = fitOLS(placed);

    // Hero
    const rate = DATA.length ? (placed.length / DATA.length) * 100 : 0;
    document.getElementById("heroTitle").textContent = `Where India's ${DATA.length.toLocaleString()} MBA graduates landed—and what drove their offers.`;
    document.getElementById("hRate").textContent = `${rate.toFixed(1)}%`;
    document.getElementById("hCounts").textContent = `${placed.length.toLocaleString()} placed out of ${DATA.length.toLocaleString()} graduates`;
    document.getElementById("hAvg").textContent = placed.length ? `₹${avg(placed.map((d) => d.PackageLPA)).toFixed(2)} LPA` : "N/A";
    document.getElementById("hMax").textContent = placed.length ? `₹${Math.max(...placed.map((d) => d.PackageLPA)).toFixed(2)} LPA` : "N/A";
    document.getElementById("hROI").textContent = placed.length ? `${avg(placed.map((d) => d.ROIMultiple)).toFixed(2)}x` : "N/A";
    document.getElementById("hStipend").textContent = DATA.length ? `₹${Math.round(avg(DATA.map((d) => d.InternshipStipendPM))).toLocaleString()}` : "N/A";
    document.getElementById("insightsSub").textContent = `Live regression and cohort comparisons across ${DATA.length.toLocaleString()} student records.`;
    document.getElementById("footerCountCopy").textContent = `Live dataset currently holds ${DATA.length.toLocaleString()} student records (${placed.length.toLocaleString()} placed).`;

    // Story Card 1: Stipend threshold
    const sBands = ["<15k", "15k-25k", "25k-50k", "50k+"];
    const ge25 = DATA.filter((d) => d.InternshipStipendPM >= 25000);
    const ge25Rate = ge25.length ? (ge25.filter((d) => d.IsPlaced === 1).length / ge25.length) * 100 : 0;
    const mid = DATA.filter((d) => d.StipendBand === "15k-25k");
    const midRate = mid.length ? (mid.filter((d) => d.IsPlaced === 1).length / mid.length) * 100 : 0;
    const low = DATA.filter((d) => d.StipendBand === "<15k");
    const lowRate = low.length ? (low.filter((d) => d.IsPlaced === 1).length / low.length) * 100 : 0;
    document.getElementById("stipendStoryCopy").innerHTML =
      `Students earning <strong>₹25,000+/month</strong> (${ge25.length.toLocaleString()} graduates) achieved a <strong>${ge25Rate.toFixed(1)}%</strong> placement rate—compared to <strong>${midRate.toFixed(1)}%</strong> in ₹15k–25k and <strong>${lowRate.toFixed(1)}%</strong> below ₹15,000.`;
    document.getElementById("stipendStoryBars").innerHTML = sBands
      .map((b) => {
        const sub = DATA.filter((d) => d.StipendBand === b);
        const pct = sub.length ? (sub.filter((d) => d.IsPlaced === 1).length / sub.length) * 100 : 0;
        return `<div class="mini-bar-row"><span>${b}</span><div class="mini-track"><div class="mini-fill" style="width:${pct.toFixed(0)}%;"></div></div><strong>${pct.toFixed(0)}%</strong></div>`;
      })
      .join("");

    // Story Card 2: Correlations & Matrix
    const rCgpa = pearson(placed.map((d) => d.MBACGPA), placed.map((d) => d.PackageLPA));
    const rExp = pearson(placed.map((d) => d.WorkExpMonths), placed.map((d) => d.PackageLPA));
    const rCat = pearson(placed.map((d) => d.CATPercentile), placed.map((d) => d.PackageLPA));
    document.getElementById("salaryStoryCopy").innerHTML =
      `<strong>MBA CGPA</strong> (r = ${rCgpa >= 0 ? "+" : ""}${rCgpa.toFixed(2)}) and <strong>Work Experience</strong> (r = ${rExp >= 0 ? "+" : ""}${rExp.toFixed(2)}) explain <strong>${(ols.r2 * 100).toFixed(1)}%</strong> of salary variation. CAT percentile (r = ${rCat.toFixed(2)}) has minimal direct impact on CTC.`;

    const lowMat = placed.filter((d) => d.CGPABand === "5.5-6.5" && d.WorkExpBand === "0m (Fresher)");
    const highMat = placed.filter((d) => d.CGPABand === "8.5-9.8" && d.WorkExpBand === "37-48m");
    document.getElementById("lowMatrixVal").textContent = lowMat.length ? `₹${avg(lowMat.map((d) => d.PackageLPA)).toFixed(2)} LPA` : "—";
    document.getElementById("highMatrixVal").textContent = highMat.length ? `₹${avg(highMat.map((d) => d.PackageLPA)).toFixed(2)} LPA` : "—";

    // B-Schools
    const colleges = uniq(DATA.map((d) => d.CollegeName));
    const schoolStats = colleges.map((name) => {
      const all = DATA.filter((d) => d.CollegeName === name);
      const pl = all.filter((d) => d.IsPlaced === 1);
      const recCounts = {};
      pl.forEach((d) => {
        recCounts[d.RecruiterCompany] = (recCounts[d.RecruiterCompany] || 0) + 1;
      });
      const topRec = Object.entries(recCounts).sort((a, b) => b[1] - a[1])[0];
      return {
        name,
        total: all.length,
        placed: pl.length,
        rate: all.length ? (pl.length / all.length) * 100 : 0,
        avgPkg: pl.length ? avg(pl.map((d) => d.PackageLPA)) : 0,
        maxPkg: pl.length ? Math.max(...pl.map((d) => d.PackageLPA)) : 0,
        avgFee: all.length ? avg(all.map((d) => d.CourseFeeLakhs)) : 0,
        roi: pl.length ? avg(pl.map((d) => d.ROIMultiple)) : 0,
        avgCat: all.length ? avg(all.map((d) => d.CATPercentile)) : 0,
        topRec: topRec ? `${topRec[0]} (${topRec[1]})` : "—",
      };
    });

    const bestRoiSchool = [...schoolStats].sort((a, b) => b.roi - a.roi)[0];
    const bestRateSchool = [...schoolStats].sort((a, b) => b.rate - a.rate)[0];
    const avgOverallRoi = placed.length ? avg(placed.map((d) => d.ROIMultiple)) : 0;
    document.getElementById("roiStoryCopy").innerHTML = bestRoiSchool
      ? `Across all campuses, 1st-year CTC averages <strong>${avgOverallRoi.toFixed(2)}x</strong> course fees. <strong>${esc(bestRoiSchool.name)}</strong> leads ROI at <strong>${bestRoiSchool.roi.toFixed(2)}x</strong> (₹${bestRoiSchool.avgPkg.toFixed(2)} LPA on ₹${bestRoiSchool.avgFee.toFixed(2)}L avg fee).`
      : "No placement records available.";
    document.getElementById("topRoiSchool").textContent = bestRoiSchool ? bestRoiSchool.name : "—";
    document.getElementById("topRateSchool").textContent = bestRateSchool ? `${bestRateSchool.name} (${bestRateSchool.rate.toFixed(1)}%)` : "—";

    const sortedSchools = [...schoolStats].sort((a, b) => {
      if (schoolSort === "pkg") return b.avgPkg - a.avgPkg;
      if (schoolSort === "rate") return b.rate - a.rate;
      return b.roi - a.roi;
    });
    document.getElementById("schoolGrid").innerHTML = sortedSchools
      .map(
        (s) => `
      <article class="school-card" data-school="${esc(s.name)}" tabindex="0">
        <div class="school-top">
          <h3>${esc(s.name)}</h3>
          <span class="roi-badge">${s.roi.toFixed(2)}x ROI</span>
        </div>
        <div class="school-metrics">
          <div><span>Avg Package</span><strong>₹${s.avgPkg.toFixed(2)} LPA</strong></div>
          <div><span>Placed Rate</span><strong>${s.rate.toFixed(1)}%</strong></div>
          <div><span>Avg Fee</span><strong>₹${s.avgFee.toFixed(2)}L</strong></div>
          <div><span>Top Recruiter</span><strong>${esc(s.topRec)}</strong></div>
        </div>
        <div class="school-foot">View campus breakdown &rarr;</div>
      </article>`
      )
      .join("");

    document.querySelectorAll(".school-card").forEach((card) => {
      const open = () => openSchoolModal(card.dataset.school, schoolStats);
      card.addEventListener("click", open);
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          open();
        }
      });
    });

    // Specializations
    const specs = uniq(DATA.map((d) => d.MBASpecialization));
    const specStats = specs
      .map((sp) => {
        const all = DATA.filter((d) => d.MBASpecialization === sp);
        const pl = all.filter((d) => d.IsPlaced === 1);
        return {
          sp,
          avgPkg: pl.length ? avg(pl.map((d) => d.PackageLPA)) : 0,
          rate: all.length ? (pl.length / all.length) * 100 : 0,
          placed: pl.length,
          stipend: all.length ? avg(all.map((d) => d.InternshipStipendPM)) : 0,
        };
      })
      .sort((a, b) => b.avgPkg - a.avgPkg);

    if (specStats.length) {
      document.getElementById("specLead").textContent = `${specStats[0].sp} currently leads average compensation at ₹${specStats[0].avgPkg.toFixed(2)} LPA (${specStats[0].rate.toFixed(1)}% placed).`;
    }
    document.getElementById("specGrid").innerHTML = specStats
      .map(
        (s) => `
      <article class="spec-card">
        <h3>${esc(s.sp)}</h3>
        <div class="spec-pkg">₹${s.avgPkg.toFixed(2)} LPA</div>
        <div class="spec-meta">${s.rate.toFixed(1)}% placed (${s.placed} offers)<br>Avg stipend: ₹${Math.round(s.stipend).toLocaleString()}/mo</div>
      </article>`
      )
      .join("");

    // Recruiters & Hubs
    const recruiters = uniq(placed.map((d) => d.RecruiterCompany))
      .map((r) => {
        const sub = placed.filter((d) => d.RecruiterCompany === r);
        return { r, hires: sub.length, avgPkg: avg(sub.map((d) => d.PackageLPA)), maxPkg: Math.max(...sub.map((d) => d.PackageLPA)) };
      })
      .sort((a, b) => b.hires - a.hires);

    const locs = uniq(placed.map((d) => d.JoiningLocation))
      .map((l) => {
        const sub = placed.filter((d) => d.JoiningLocation === l);
        return { l, hires: sub.length, avgPkg: avg(sub.map((d) => d.PackageLPA)) };
      })
      .sort((a, b) => b.hires - a.hires);

    document.getElementById("recLead").textContent = `${recruiters.length} recruiting firms hiring across ${locs.length} Indian metro hubs.`;
    document.getElementById("recruiterGrid").innerHTML = recruiters
      .map(
        (x) => `
      <article class="rec-card">
        <h4>${esc(x.r)}</h4>
        <div class="rec-row"><span>Total Hires</span><strong>${x.hires}</strong></div>
        <div class="rec-row"><span>Avg Package</span><strong>₹${x.avgPkg.toFixed(2)} LPA</strong></div>
        <div class="rec-row"><span>Highest Offer</span><strong>₹${x.maxPkg.toFixed(2)} LPA</strong></div>
      </article>`
      )
      .join("");

    document.getElementById("hubStrip").innerHTML = locs
      .map((h) => `<div class="hub-item"><span>${esc(h.l)} Hub</span><strong>${h.hires} hires &bull; ₹${h.avgPkg.toFixed(2)}L</strong></div>`)
      .join("");

    updateCalculator(ols, placed);
    renderDirectory();
    renderAdminTable();
  }

  function openSchoolModal(name, schoolStats) {
    const s = schoolStats.find((x) => x.name === name);
    if (!s) return;
    const pl = DATA.filter((d) => d.IsPlaced === 1 && d.CollegeName === name);
    const specs = uniq(DATA.map((d) => d.MBASpecialization));
    const specRows = specs
      .map((sp) => {
        const sub = pl.filter((d) => d.MBASpecialization === sp);
        return `<li><strong>${esc(sp)}:</strong> ${sub.length} placed &bull; ₹${sub.length ? avg(sub.map((x) => x.PackageLPA)).toFixed(2) : "0.00"} LPA avg</li>`;
      })
      .join("");
    const topOffers = [...pl]
      .sort((a, b) => b.PackageLPA - a.PackageLPA)
      .slice(0, 4)
      .map((o) => `<li><strong>${esc(o.StudentName)}</strong> (${esc(o.MBASpecialization)}) &rarr; ${esc(o.RecruiterCompany)} at <strong>₹${o.PackageLPA.toFixed(2)} LPA</strong></li>`)
      .join("");

    document.getElementById("modalBody").innerHTML = `
      <span class="eyebrow">Campus Profile</span>
      <h2 id="modalTitle">${esc(s.name)}</h2>
      <p class="lead-sm">${s.placed} placed out of ${s.total} students (${s.rate.toFixed(1)}%) &bull; Avg CAT: ${s.avgCat.toFixed(1)} percentile</p>
      <div class="modal-grid">
        <div class="modal-box"><span>Average Package</span><strong>₹${s.avgPkg.toFixed(2)} LPA</strong></div>
        <div class="modal-box"><span>Highest Package</span><strong>₹${s.maxPkg.toFixed(2)} LPA</strong></div>
        <div class="modal-box"><span>Average Fee</span><strong>₹${s.avgFee.toFixed(2)} Lakhs</strong></div>
        <div class="modal-box"><span>1st-Year ROI</span><strong>${s.roi.toFixed(2)}x</strong></div>
      </div>
      <h4>Specialization Averages</h4>
      <ul style="margin:8px 0 16px 18px;font-size:14px;color:#334155;">${specRows}</ul>
      <h4>Top Offers</h4>
      <ul style="margin:8px 0 0 18px;font-size:14px;color:#334155;">${topOffers}</ul>
    `;
    const modal = document.getElementById("schoolModal");
    modal.classList.add("open");
    modal.setAttribute("aria-hidden", "false");
  }

  function updateCalculator(olsOverride, placedOverride) {
    const placed = placedOverride || DATA.filter((d) => d.IsPlaced === 1);
    const ols = olsOverride || fitOLS(placed);

    const school = document.getElementById("cSchool").value;
    const spec = document.getElementById("cSpec").value;
    const cgpa = parseFloat(document.getElementById("cCGPA").value);
    const exp = parseInt(document.getElementById("cExp").value, 10);
    const stipend = parseInt(document.getElementById("cStipend").value, 10);
    const fee = parseFloat(document.getElementById("cFee").value);

    document.getElementById("cCGPAVal").textContent = cgpa.toFixed(2);
    document.getElementById("cExpVal").textContent = `${exp} months`;
    document.getElementById("cStipendVal").textContent = `₹${stipend.toLocaleString()} / mo`;
    document.getElementById("calcModelNote").textContent = `Live OLS: Package = ${ols.intercept.toFixed(2)} + ${ols.cgpaCoef.toFixed(2)}×CGPA + ${ols.expCoef.toFixed(3)}×Exp (R² = ${ols.r2.toFixed(3)}, RMSE = ${ols.rmse.toFixed(2)} LPA)`;

    const est = ols.intercept + ols.cgpaCoef * cgpa + ols.expCoef * exp;
    const low = Math.max(0, est - ols.rmse);
    const high = est + ols.rmse;
    const band = stipend < 15000 ? "<15k" : stipend < 25000 ? "15k-25k" : stipend < 50000 ? "25k-50k" : "50k+";
    const bandRows = DATA.filter((d) => d.StipendBand === band);
    const outlook = bandRows.length ? (bandRows.filter((d) => d.IsPlaced === 1).length / bandRows.length) * 100 : 0;
    const roi = est / fee;
    const payback = est > 0 ? (fee / est) * 12 : 0;
    const pct = placed.length ? (placed.filter((d) => d.PackageLPA <= est).length / placed.length) * 100 : 0;

    document.getElementById("rPkg").textContent = `₹${est.toFixed(2)} LPA`;
    document.getElementById("rRange").textContent = `Typical range: ₹${low.toFixed(1)} – ₹${high.toFixed(1)} LPA (±1 RMSE)`;
    document.getElementById("rOutlook").textContent = `${outlook.toFixed(1)}%`;
    document.getElementById("rROI").textContent = `${roi.toFixed(2)}x`;
    document.getElementById("rPayback").textContent = `${payback.toFixed(1)} mos`;
    document.getElementById("rPct").textContent = `${Math.round(pct)}th`;

    const peers = placed
      .map((d) => ({
        d,
        dist:
          Math.abs(d.MBACGPA - cgpa) * 3 +
          Math.abs(d.WorkExpMonths - exp) * 0.15 +
          (d.CollegeName === school ? 0 : 0.7) +
          (d.MBASpecialization === spec ? 0 : 0.7),
      }))
      .sort((a, b) => a.dist - b.dist)
      .slice(0, 4)
      .map((x) => x.d);

    document.getElementById("rPeers").innerHTML = peers
      .map(
        (p) => `
      <div class="peer-item">
        <span><strong>${esc(p.StudentName)}</strong> &bull; ${esc(p.CollegeName)} (${p.MBACGPA.toFixed(2)} CGPA, ${p.WorkExpMonths}m)</span>
        <strong>${esc(p.RecruiterCompany)} &bull; ₹${p.PackageLPA.toFixed(2)}L</strong>
      </div>`
      )
      .join("");
  }

  function renderDirectory() {
    const q = document.getElementById("dirSearch").value.trim().toLowerCase();
    const sc = document.getElementById("dirSchool").value;
    const sp = document.getElementById("dirSpec").value;
    const out = document.getElementById("dirOutcome").value;

    const filtered = DATA.filter((d) => {
      if (sc !== "ALL" && d.CollegeName !== sc) return false;
      if (sp !== "ALL" && d.MBASpecialization !== sp) return false;
      if (out === "PLACED" && d.IsPlaced !== 1) return false;
      if (out === "UNPLACED" && d.IsPlaced !== 0) return false;
      if (!q) return true;
      return (
        d.StudentName.toLowerCase().includes(q) ||
        d.StudentID.toLowerCase().includes(q) ||
        d.CollegeName.toLowerCase().includes(q) ||
        (d.RecruiterCompany && d.RecruiterCompany.toLowerCase().includes(q)) ||
        (d.JoiningLocation && d.JoiningLocation.toLowerCase().includes(q))
      );
    });

    // Show newest added records first if IDs > 4000 or reverse? Keep natural order with newest first when added
    const ordered = [...filtered].reverse();
    const shown = ordered.slice(0, visibleCount);
    document.getElementById("dirStatus").textContent = `Showing ${shown.length.toLocaleString()} of ${filtered.length.toLocaleString()} graduates (newest first)`;
    document.getElementById("loadMoreBtn").style.display = visibleCount < filtered.length ? "inline-flex" : "none";

    document.getElementById("gradGrid").innerHTML = shown
      .map((d) => {
        const badge = d.IsPlaced
          ? '<span class="badge badge-placed">Placed</span>'
          : '<span class="badge badge-unplaced">Opted Out / Unplaced</span>';
        const offerLine = d.IsPlaced
          ? `<span>${esc(d.RecruiterCompany)} &bull; ${esc(d.JoiningLocation)}</span><strong>₹${d.PackageLPA.toFixed(2)} LPA</strong>`
          : `<span>Internship: ₹${d.InternshipStipendPM.toLocaleString()}/mo</span><strong>No Offer</strong>`;
        return `
        <article class="grad-card">
          <div class="grad-head">
            <h4>${esc(d.StudentName)}</h4>
            ${badge}
          </div>
          <div class="grad-sub">${esc(d.StudentID)} &bull; ${esc(d.CollegeName)} &bull; ${esc(d.MBASpecialization)}</div>
          <div class="grad-sub">CGPA: <strong>${d.MBACGPA.toFixed(2)}</strong> &bull; Exp: <strong>${d.WorkExpMonths}m</strong> &bull; ${esc(d.UndergradDegree)}</div>
          <div class="grad-offer">${offerLine}</div>
        </article>`;
      })
      .join("");
  }

  // Admin Panel logic
  function showAdminMsg(text, isError) {
    const el = document.getElementById("adminMsg");
    el.textContent = text;
    el.className = "admin-msg " + (isError ? "err" : "ok");
  }

  function renderAdminTable() {
    const q = document.getElementById("adminSearch").value.trim().toLowerCase();
    const rows = [...DATA]
      .reverse()
      .filter((d) => {
        if (!q) return true;
        return (
          d.StudentID.toLowerCase().includes(q) ||
          d.StudentName.toLowerCase().includes(q) ||
          d.CollegeName.toLowerCase().includes(q) ||
          (d.RecruiterCompany && d.RecruiterCompany.toLowerCase().includes(q))
        );
      })
      .slice(0, 100);

    document.getElementById("adminTableBody").innerHTML = rows
      .map((d) => {
        const checked = selectedIds.has(d.StudentID) ? "checked" : "";
        return `<tr>
          <td><input type="checkbox" class="row-check" data-id="${esc(d.StudentID)}" ${checked} /></td>
          <td>${esc(d.StudentID)}</td>
          <td><strong>${esc(d.StudentName)}</strong></td>
          <td>${esc(d.CollegeName)}</td>
          <td>${esc(d.MBASpecialization)}</td>
          <td>${esc(d.EmploymentStatus)}</td>
          <td>${esc(d.RecruiterCompany || "—")}</td>
          <td class="num">${d.PackageLPA !== null ? d.PackageLPA.toFixed(2) : "—"}</td>
          <td><button type="button" class="btn-sm btn-danger-outline row-del" data-id="${esc(d.StudentID)}">Remove</button></td>
        </tr>`;
      })
      .join("");

    document.querySelectorAll(".row-check").forEach((cb) => {
      cb.addEventListener("change", () => {
        if (cb.checked) selectedIds.add(cb.dataset.id);
        else selectedIds.delete(cb.dataset.id);
        updateBulkBtn();
      });
    });

    document.querySelectorAll(".row-del").forEach((btn) => {
      btn.addEventListener("click", () => handleDelete([btn.dataset.id]));
    });
    updateBulkBtn();
  }

  function updateBulkBtn() {
    const btn = document.getElementById("bulkDeleteBtn");
    btn.textContent = `Delete Selected (${selectedIds.size})`;
    btn.disabled = selectedIds.size === 0;
  }

  async function handleDelete(ids) {
    if (!ids.length) return;
    if (serverOnline) {
      const res = await fetch("/api/students/bulk-delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ studentIds: ids }),
      });
      const json = await res.json();
      if (!res.ok) return showAdminMsg(json.error || "Delete failed", true);
    }
    const idSet = new Set(ids);
    DATA = DATA.filter((d) => !idSet.has(d.StudentID));
    ids.forEach((id) => selectedIds.delete(id));
    if (!serverOnline) persistLocal();
    renderAll();
    document.getElementById("aID").value = nextStudentId();
    showAdminMsg(`Removed ${ids.length} record(s). All metrics and directories synced.`, false);
  }

  document.getElementById("addStudentForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const isPlaced = document.getElementById("aStatus").value === "Placed";
    const workExp = parseInt(document.getElementById("aExp").value, 10) || 0;
    const payload = {
      StudentID: document.getElementById("aID").value.trim(),
      StudentName: document.getElementById("aName").value.trim(),
      Gender: document.getElementById("aGender").value,
      Age: parseInt(document.getElementById("aAge").value, 10),
      HomeState: document.getElementById("aState").value,
      CollegeName: document.getElementById("aCollege").value,
      MBASpecialization: document.getElementById("aSpec").value,
      UndergradDegree: document.getElementById("aUg").value,
      UndergradPercentage: parseFloat(document.getElementById("aUgPct").value),
      Class10Score: parseFloat(document.getElementById("a10").value),
      Class12Score: parseFloat(document.getElementById("a12").value),
      CATPercentile: parseFloat(document.getElementById("aCat").value),
      MBACGPA: parseFloat(document.getElementById("aCGPA").value),
      WorkExpMonths: workExp,
      PriorWorkSector: workExp === 0 ? "No Experience" : document.getElementById("aSector").value,
      EmploymentStatus: document.getElementById("aStatus").value,
      RecruiterCompany: isPlaced ? document.getElementById("aRecruiter").value : null,
      PlacementType: isPlaced ? document.getElementById("aPType").value : null,
      PackageLPA: isPlaced ? parseFloat(document.getElementById("aPkg").value) : null,
      JoiningLocation: isPlaced ? document.getElementById("aLoc").value : null,
      InternshipStipendPM: parseInt(document.getElementById("aStipend").value, 10),
      CertificationsCount: 2,
      LiveProjectsCompleted: 2,
      SoftSkillsScore: 7,
      CourseFeeINR: parseInt(document.getElementById("aFee").value, 10),
    };

    if (DATA.some((d) => d.StudentID === payload.StudentID)) {
      return showAdminMsg(`StudentID "${payload.StudentID}" already exists.`, true);
    }
    if (isPlaced && (!payload.PackageLPA || payload.PackageLPA <= 0)) {
      return showAdminMsg("Placed students must have a Package (LPA) greater than 0.", true);
    }

    if (serverOnline) {
      const res = await fetch("/api/students", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const json = await res.json();
      if (!res.ok) return showAdminMsg(json.error || "Failed to add student", true);
    }

    DATA.push(enrichRow(payload));
    if (!serverOnline) persistLocal();
    renderAll();
    document.getElementById("aName").value = "";
    document.getElementById("aID").value = nextStudentId();
    showAdminMsg(`Added ${payload.StudentName} (${payload.StudentID}). Everything synced across all views.`, false);
  });

  // Form dynamic toggles
  document.getElementById("aStatus").addEventListener("change", () => {
    const placed = document.getElementById("aStatus").value === "Placed";
    document.querySelectorAll(".placement-only").forEach((el) => el.classList.toggle("disabled", !placed));
  });
  document.getElementById("aExp").addEventListener("input", () => {
    const exp = parseInt(document.getElementById("aExp").value, 10) || 0;
    const sec = document.getElementById("aSector");
    if (exp === 0) sec.value = "No Experience";
    else if (sec.value === "No Experience") sec.value = "IT/ITeS";
  });

  // Admin modal controls
  document.getElementById("openAdminBtn").addEventListener("click", () => {
    document.getElementById("aID").value = nextStudentId();
    renderAdminTable();
    const m = document.getElementById("adminModal");
    m.classList.add("open");
    m.setAttribute("aria-hidden", "false");
  });
  document.getElementById("adminClose").addEventListener("click", () => {
    const m = document.getElementById("adminModal");
    m.classList.remove("open");
    m.setAttribute("aria-hidden", "true");
  });
  document.getElementById("adminModal").addEventListener("click", (e) => {
    if (e.target.id === "adminModal") {
      e.target.classList.remove("open");
      e.target.setAttribute("aria-hidden", "true");
    }
  });

  document.querySelectorAll("[data-atab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-atab]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const isAdd = btn.dataset.atab === "add";
      document.getElementById("addStudentForm").style.display = isAdd ? "block" : "none";
      document.getElementById("manageStudentPanel").style.display = isAdd ? "none" : "block";
    });
  });

  document.getElementById("adminSearch").addEventListener("input", renderAdminTable);
  document.getElementById("selectAllRows").addEventListener("change", (e) => {
    document.querySelectorAll(".row-check").forEach((cb) => {
      cb.checked = e.target.checked;
      if (cb.checked) selectedIds.add(cb.dataset.id);
      else selectedIds.delete(cb.dataset.id);
    });
    updateBulkBtn();
  });
  document.getElementById("bulkDeleteBtn").addEventListener("click", () => handleDelete([...selectedIds]));

  document.getElementById("resetDataBtn").addEventListener("click", async () => {
    if (serverOnline) {
      const res = await fetch("/api/reset", { method: "POST" });
      if (!res.ok) return showAdminMsg("Server reset failed", true);
    }
    localStorage.removeItem(STORAGE_KEY);
    DATA = BASE_DATA.map(enrichRow);
    selectedIds.clear();
    renderAll();
    document.getElementById("aID").value = nextStudentId();
    showAdminMsg("Restored baseline 3,000 cleaned records.", false);
  });

  document.getElementById("exportCsvBtn").addEventListener("click", () => {
    const cols = [
      "StudentID", "StudentName", "Gender", "Age", "HomeState", "CollegeName",
      "MBASpecialization", "UndergradDegree", "UndergradPercentage", "Class10Score",
      "Class12Score", "CATPercentile", "MBACGPA", "WorkExpMonths", "PriorWorkSector",
      "EmploymentStatus", "RecruiterCompany", "PlacementType", "PackageLPA",
      "JoiningLocation", "InternshipStipendPM", "CertificationsCount",
      "LiveProjectsCompleted", "SoftSkillsScore", "CourseFeeINR",
    ];
    const lines = [
      cols.join(","),
      ...DATA.map((d) =>
        cols
          .map((c) => {
            const val = c === "CourseFeeINR" ? Math.round((d.CourseFeeLakhs || 0) * 100000) : d[c];
            const s = val === null || val === undefined ? "" : String(val);
            return s.includes(",") ? `"${s.replace(/"/g, '""')}"` : s;
          })
          .join(",")
      ),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "cleaned_data_synced.csv";
    a.click();
    URL.revokeObjectURL(url);
  });

  // Existing listeners
  document.querySelectorAll(".pill-btn[data-sort]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".pill-btn[data-sort]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      schoolSort = btn.dataset.sort;
      renderAll();
    });
  });

  document.getElementById("modalClose").addEventListener("click", () => {
    const modal = document.getElementById("schoolModal");
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
  });
  document.getElementById("schoolModal").addEventListener("click", (e) => {
    if (e.target.id === "schoolModal") {
      e.target.classList.remove("open");
      e.target.setAttribute("aria-hidden", "true");
    }
  });

  ["cSchool", "cSpec", "cCGPA", "cExp", "cStipend", "cFee"].forEach((id) => {
    document.getElementById(id).addEventListener("input", () => updateCalculator());
  });
  ["dirSearch", "dirSchool", "dirSpec", "dirOutcome"].forEach((id) => {
    document.getElementById(id).addEventListener("input", () => {
      visibleCount = 12;
      renderDirectory();
    });
  });
  document.getElementById("loadMoreBtn").addEventListener("click", () => {
    visibleCount += 12;
    renderDirectory();
  });

  async function boot() {
    const pill = document.getElementById("syncPill");
    if (window.location.protocol.startsWith("http")) {
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        if (res.ok) serverOnline = true;
      } catch (_) {}
    }
    initDataFromStorageOrBase();
    if (serverOnline) {
      pill.textContent = "Server + CSV Synced";
      pill.className = "sync-pill live";
      document.getElementById("adminSyncDesc").textContent =
        "Connected to server.py: adding or deleting records writes directly to cleaned_data.csv, Power BI tables, and website/data.js.";
    } else {
      pill.textContent = "Browser Storage Sync";
      pill.className = "sync-pill local";
      document.getElementById("adminSyncDesc").textContent =
        "Running standalone (file://): edits persist in browser localStorage and sync across the whole page. Start server.py to also write edits back to CSV files.";
    }
    renderAll();
  }

  boot();
})();
