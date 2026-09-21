const benchmarks = {
  odir: {name:'ODIR-5K',note:'Binocular fundus images with 8 disease labels.',rows:[[.502,.518,.522,.707,.711,.718],[.501,.506,.524,.432,.443,.436],[.449,.487,.488,.351,.385,.401],[.531,.527,.542,.390,.393,.407],[.542,.528,.525,.813,.819,.931],[.665,.669,.690,.888,.889,.901],[.936,.939,.937,.989,.971,.972]]},
  mimic: {name:'MIMIC-CXR',note:'Large-scale chest radiographs and reports with a long-tailed label distribution.',rows:[[.681,.693,.697,.713,.720,.723],[.667,.669,.668,.616,.620,.621],[.637,.671,.682,.591,.620,.629],[.668,.672,.669,.624,.632,.627],[.690,.693,.699,.841,.846,.876],[.686,.689,.699,.749,.756,.763],[.808,.823,.830,.868,.898,.901]]},
  iu: {name:'IU-Xray',note:'Chest X-rays paired with clinical reports.',rows:[[.741,.746,.761,.836,.836,.836],[.721,.735,.740,.686,.685,.683],[.719,.721,.723,.637,.673,.675],[.713,.715,.717,.660,.655,.655],[.745,.753,.754,.800,.778,.759],[null,null,null,null,null,null],[.894,.895,.899,.892,.899,.896]]}
};
const methods = ['DCHMT','MITH','DSPH','CMCL','CICH','MSACH','TriPAH'];
document.querySelectorAll('[data-dataset]').forEach(button => {
  button.addEventListener('click', () => {
    const data = benchmarks[button.dataset.dataset];
    document.querySelectorAll('[data-dataset]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    const rows = data.rows.map((values, index) => {
      const row = document.createElement('tr');
      if(index === 6) row.className = 'ours';
      const label = document.createElement('th'); label.scope = 'row'; label.textContent = methods[index];
      if(index === 6){ const badge=document.createElement('span'); badge.textContent='Ours'; label.append(' ',badge); }
      row.append(label);
      values.forEach(value => {const cell=document.createElement('td'); cell.textContent=value === null ? '—' : value.toFixed(3);row.append(cell);});
      return row;
    });
    document.getElementById('benchmark-body').replaceChildren(...rows);
    document.getElementById('benchmark-caption').textContent = `${data.name} · Mean average precision (mAP ↑)`;
    document.getElementById('benchmark-status').textContent = `${data.name}: ${data.note}`;
  });
});
