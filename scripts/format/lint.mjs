import markdownlint from 'markdownlint';
import markdownlintRuleHelpers from 'markdownlint/helpers';
import { readFile } from 'fs/promises';
import jsonc from 'jsonc-parser';

const configPath = '.markdownlint.jsonc';

export default async function lint(content) {
  let config = await readFile(configPath, 'utf8');
  const fixResult = markdownlint.sync({
    strings: { content: content },
    config: jsonc.parse(config),
  });
  const fixes = fixResult['content'].filter((error) => error.fixInfo);
  if (fixes.length > 0) {
    const fixedText = markdownlintRuleHelpers.applyFixes(content, fixes);
    return fixedText;
  }
  return content;
}
