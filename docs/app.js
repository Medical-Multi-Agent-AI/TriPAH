const benchmarks = {
  odir: {name:'ODIR-5K',note:'Binocular fundus images with 8 disease labels.',rows:[[.502,.518,.522,.707,.711,.718],[.501,.506,.524,.432,.443,.436],[.449,.487,.488,.351,.385,.401],[.531,.527,.542,.390,.393,.407],[.542,.528,.525,.813,.819,.931],[.665,.669,.690,.888,.889,.901],[.936,.939,.937,.989,.971,.972]]},
  mimic: {name:'MIMIC-CXR',note:'Large-scale chest radiographs and reports with a long-tailed label distribution.',rows:[[.681,.693,.697,.713,.720,.723],[.667,.669,.668,.616,.620,.621],[.637,.671,.682,.591,.620,.629],[.668,.672,.669,.624,.632,.627],[.690,.693,.699,.841,.846,.876],[.686,.689,.699,.749,.756,.763],[.808,.823,.830,.868,.898,.901]]},
  iu: {name:'IU-Xray',note:'Chest X-rays paired with clinical reports.',rows:[[.741,.746,.761,.836,.836,.836],[.721,.735,.740,.686,.685,.683],[.719,.721,.723,.637,.673,.675],[.713,.715,.717,.660,.655,.655],[.745,.753,.754,.800,.778,.759],[null,null,null,null,null,null],[.894,.895,.899,.892,.899,.896]]}
};
const methods = ['DCHMT','MITH','DSPH','CMCL','CICH','MSACH','TriPAH'];
function renderBenchmark(key) {
    const data = benchmarks[key];
    const ranks = Array.from({length:6}, (_, column) => [...new Set(data.rows.map(row => row[column]).filter(value => value !== null))].sort((a,b) => b-a));
    document.querySelectorAll('[data-dataset]').forEach(item => item.setAttribute('aria-pressed', String(item.dataset.dataset === key)));
    const rows = data.rows.map((values, index) => {
      const row = document.createElement('tr');
      if(index === 6) row.className = 'ours';
      const label = document.createElement('th'); label.scope = 'row'; label.textContent = methods[index];
      if(index === 6){ const badge=document.createElement('span'); badge.textContent='Ours'; label.append(' ',badge); }
      row.append(label);
      values.forEach((value, column) => {
        const cell = document.createElement('td');
        cell.textContent = value === null ? '—' : value.toFixed(3);
        if (value !== null && value === ranks[column][0]) {
          cell.className = 'best';
          const label = document.createElement('span'); label.className = 'sr-only'; label.textContent = ' (best)'; cell.append(label);
        } else if (value !== null && value === ranks[column][1]) {
          cell.className = 'runner-up';
          const label = document.createElement('span'); label.className = 'sr-only'; label.textContent = ' (second best)'; cell.append(label);
        }
        row.append(cell);
      });
      return row;
    });
    document.getElementById('benchmark-body').replaceChildren(...rows);
    document.getElementById('benchmark-caption').textContent = `${data.name} · Mean average precision (mAP ↑)`;
    document.getElementById('benchmark-status').textContent = `${data.name}: ${data.note}`;
}
document.querySelectorAll('[data-dataset]').forEach(button => {
  button.addEventListener('click', () => renderBenchmark(button.dataset.dataset));
});
renderBenchmark('odir');

const carousel = document.querySelector('.carousel');
if (carousel) {
  const slides = [...carousel.querySelectorAll('.slide')];
  const dots = [...carousel.querySelectorAll('[data-slide]')];
  let currentSlide = 0;
  function showSlide(index) {
    currentSlide = (index + slides.length) % slides.length;
    slides.forEach((slide, i) => { slide.hidden = i !== currentSlide; });
    dots.forEach((dot, i) => dot.setAttribute('aria-pressed', String(i === currentSlide)));
    carousel.querySelector('.slide-count').textContent = `${currentSlide + 1} / ${slides.length}`;
    carousel.querySelector('.gallery-open').href = slides[currentSlide].querySelector('.slide-visual').href;
    document.getElementById('slide-status').textContent = slides[currentSlide].getAttribute('aria-label');
  }
  carousel.querySelector('[data-slide-prev]').addEventListener('click', () => showSlide(currentSlide - 1));
  carousel.querySelector('[data-slide-next]').addEventListener('click', () => showSlide(currentSlide + 1));
  dots.forEach(dot => dot.addEventListener('click', () => showSlide(Number(dot.dataset.slide))));
  carousel.addEventListener('keydown', event => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      showSlide(currentSlide + (event.key === 'ArrowRight' ? 1 : -1));
    }
  });
  let touchStart = null;
  let suppressClick = false;
  carousel.addEventListener('pointerdown', event => {
    suppressClick = false;
    if (event.pointerType === 'touch' && event.target.closest('.slide-visual') && !event.target.closest('.slide-qualitative')) {
      touchStart = {x:event.clientX, y:event.clientY};
    }
  });
  carousel.addEventListener('pointerup', event => {
    if (!touchStart) return;
    const dx = event.clientX - touchStart.x;
    const dy = event.clientY - touchStart.y;
    touchStart = null;
    if (Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy) * 1.5) {
      suppressClick = true;
      showSlide(currentSlide + (dx < 0 ? 1 : -1));
    }
  });
  carousel.addEventListener('pointercancel', () => { touchStart = null; });
  carousel.addEventListener('click', event => {
    if (suppressClick && event.target.closest('.slide-visual')) {
      event.preventDefault();
      suppressClick = false;
    }
  }, true);
}
