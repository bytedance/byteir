// check markdown spacing between Chinese and English characters

import autocorrect from 'autocorrect-node';

export default function space(content) {
  return autocorrect.format(content);
}
