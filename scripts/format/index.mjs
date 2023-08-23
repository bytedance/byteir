import { readdir, readFile, writeFile } from 'fs/promises';
import { join, extname } from 'path';
import chalk from 'chalk';
import lint from './lint.mjs';
import space from './space.mjs';

async function main() {
  // chinese doc use markdownlint and autocorrect to format
  let zhChanged = await formatDocs('./content/zh/docs/hertz', [
    function (content) {
      return space(content);
    },
  ]);

  // english doc use markdownlint to format
  let enChanged = await formatDocs('./content/en/docs/hertz');

  if (zhChanged || enChanged) {
    console.error(
      chalk.red(`\nProper commits require formatted documentation.`) +
        chalk.red(
          `\nThe document has been automatically repaired, please check whether the formatted document is as expected.`
        )
    );
    process.exit(1);
  }
}

async function formatDocs(dir, extensionFn = []) {
  let changed = false;
  const mdPaths = await getPaths(dir, '.md', ['_index.md']);
  for (const mdPath of mdPaths) {
    let content = await readFile(mdPath, 'utf8');
    let formatContent = content;
    // fix space
    for (const fn of extensionFn) {
      formatContent = fn(content);
    }
    // fix lint
    formatContent = await lint(formatContent);
    if (content != formatContent) {
      console.info('fixing file', mdPath);
      await writeFile(mdPath, formatContent, {
        encoding: 'utf8',
        flag: 'w',
      });
      changed = true;
    }
  }
  return changed;
}

async function getPaths(currentDirPath, ext = '', excludes = []) {
  let mdPaths = [];
  const dirents = await readdir(currentDirPath, {
    withFileTypes: true,
  });
  for (const dirent of dirents) {
    const cpath = join(currentDirPath, dirent.name);
    if (dirent.isFile()) {
      if (extname(dirent.name) == ext) {
        if (!excludes.includes(dirent.name)) {
          mdPaths.push(cpath);
        }
      }
    } else if (dirent.isDirectory()) {
      let subMdPaths = await getPaths(cpath, ext, excludes);
      mdPaths.push(...subMdPaths);
    }
  }
  return mdPaths;
}

main();
