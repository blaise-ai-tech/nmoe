import { Window } from 'happy-dom'

const win = new Window()
Object.assign(globalThis, {
  window: win.window,
  document: win.document,
  navigator: win.navigator,
  HTMLElement: (win.window as any).HTMLElement,
})

