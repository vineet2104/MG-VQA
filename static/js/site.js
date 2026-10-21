'use strict';
(() => {
  const examples = window.MGVQA_EXAMPLES || [];
  const video = document.getElementById('example-video');
  if (!video || !examples.length) return;
  const tabs = [...document.querySelectorAll('[role="tab"][data-example]')];
  const panel = document.getElementById('example-panel');
  const observation = document.getElementById('example-observation');
  const play = document.getElementById('example-play');
  const loop = document.getElementById('example-loop');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let active = examples[0];
  let activeIndex = 0;
  let currentTurn = null;
  let frameCallback = null;
  let generation = 0;

  function update(time = video.currentTime) {
    const segment = active.segments.find(s => time >= s.start && time < s.end)
      || (time >= active.duration ? active.segments.at(-1) : active.segments[0]);
    if (!segment || currentTurn === segment.turn) return;
    currentTurn = segment.turn;
    panel.dataset.turn = String(segment.turn);
    panel.dataset.overlay = segment.overlay;
    observation.src = segment.image;
    observation.alt = `Top-down observation for ${active.task}, turn ${segment.turn}: ${segment.label}`;
    document.getElementById('example-tool').textContent = segment.label;
    document.getElementById('example-reason').textContent = segment.reason;
  }

  function startFrameUpdates() {
    if (!('requestVideoFrameCallback' in video)) return;
    if (frameCallback !== null) video.cancelVideoFrameCallback(frameCallback);
    const version = generation;
    const frame = (_, metadata) => {
      if (version !== generation) return;
      if (!video.seeking) update(metadata.mediaTime);
      frameCallback = video.requestVideoFrameCallback(frame);
    };
    frameCallback = video.requestVideoFrameCallback(frame);
  }

  function tryPlay() {
    const pending = video.play();
    if (pending) pending.catch(() => { play.textContent = video.paused ? 'Play' : 'Pause'; });
  }

  function select(index, focus = false) {
    generation += 1;
    if (frameCallback !== null && 'cancelVideoFrameCallback' in video) {
      video.cancelVideoFrameCallback(frameCallback);
      frameCallback = null;
    }
    video.pause();
    activeIndex = index;
    active = examples[index];
    currentTurn = null;
    panel.dataset.episode = String(active.episode);
    panel.setAttribute('aria-labelledby', tabs[index].id);
    tabs.forEach((tab, i) => {
      tab.setAttribute('aria-selected', String(i === index));
      tab.tabIndex = i === index ? 0 : -1;
    });
    if (focus) tabs[index].focus();
    document.getElementById('example-question').textContent = active.question;
    const options = document.getElementById('example-options');
    options.replaceChildren();
    active.options.forEach((option, i) => {
      const li = document.createElement('li');
      li.textContent = /^[A-E](?::|$)/.test(option) ? option : `${String.fromCharCode(65 + i)}: ${option}`;
      options.append(li);
    });
    options.hidden = active.options.length === 0;
    document.getElementById('example-open-answer').hidden = active.options.length > 0;
    const badge = document.getElementById('example-outcome');
    badge.classList.toggle('correct', active.correct);
    badge.classList.toggle('incorrect', !active.correct);
    document.getElementById('example-response').textContent = `${active.response} · ${active.correct ? 'Correct' : 'Incorrect'}`;
    const correction = document.getElementById('example-correction');
    correction.hidden = active.correct;
    correction.textContent = active.correct ? '' : `Correct: ${active.correctOption} — ${active.correctText}`;
    document.getElementById('example-caption').textContent = active.caption;
    video.poster = active.poster;
    video.src = active.video;
    video.muted = true;
    video.loop = loop.checked;
    video.autoplay = !reducedMotion;
    video.load();
    update(0);
    // Warm only this example's observations, avoiding an image flash at tool boundaries.
    active.segments.forEach(segment => { const image = new Image(); image.src = segment.image; });
    if (!reducedMotion) tryPlay();
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => select(index));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (activeIndex + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') next = (activeIndex + tabs.length - 1) % tabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = tabs.length - 1;
      else return;
      event.preventDefault();
      select(next, true);
    });
  });
  play.addEventListener('click', () => video.paused ? tryPlay() : video.pause());
  loop.addEventListener('change', () => { video.loop = loop.checked; });
  video.addEventListener('play', () => { play.textContent = 'Pause'; });
  video.addEventListener('pause', () => { play.textContent = 'Play'; });
  video.addEventListener('ended', () => { play.textContent = 'Play'; update(); });
  video.addEventListener('loadeddata', startFrameUpdates);
  for (const event of ['loadedmetadata', 'timeupdate', 'seeking', 'seeked']) {
    video.addEventListener(event, () => update());
  }
  document.addEventListener('visibilitychange', () => { if (document.hidden) video.pause(); });
  select(0);
})();
