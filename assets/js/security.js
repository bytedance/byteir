/*
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
(function($){
    $.safetyData = [{
        id: 1,
        bulletin: 'Project_Name-SA-2022-1',
        digest: 'Connection Leaking',
        influence: ['kitex-v0.1.3'],
        level: 'Low',
        publish: '2022-01-13'
    },{
        id: 2,
        bulletin: 'Project_Name-SA-2022-2',
        digest: 'Netpoll Panic',
        influence: ['netpoll-v0.2.2', 'kitex-v0.3.0'],
        level: 'Middle',
        publish: '2022-05-09'
    }];
})(jQuery);

(function($) {
    'use strict';
    function renderSafetyTable($, data) {
        const $tbody = $('#bulletinTable > tbody');
        const isCN = location.href.indexOf('zh') > -1 ? '/zh' : '';
        let html = '';
        if (data.length === 0) {
            $tbody.html('<tr><td colspan="5">&nbsp;</td></tr>');
            return;
        }
        for(let i =0;i< data.length;i++){
            const item = data[i];
            const bulletinList = [
                '<tr><td><a href=\''+ isCN + '/security/safety-bulletin/detail/'+ item.bulletin.toLowerCase() + '/\'>'+ item.bulletin + '</a>',
                '<td>'+ item.digest + '</td>',
                '<td>'+ item.level + '</td>',
                '<td>'+ item.influence.join('<br/>') + '</td>',
                '<td>'+ item.publish + '</td></tr>',
            ].join('\n');
            html+= bulletinList;
        }
        $tbody.html(html);
    }

    function filterData(source, keyword, level, year) {
        const compose = (arr) => {
            return arr.reduce(
              (prev, next) =>
                (...args) =>
                  prev(next(...args)),
            );
        };
        const rules = {
            getAll: (data) => data,
            getWord: (data) => data.filter((item)=> {
                return item.bulletin.indexOf(keyword) > -1 || (item.digest.indexOf(keyword)) > -1 || (item.influence.join('').indexOf(keyword)) > -1;
            }),
            getLevel: (data) => data.filter((item)=> item.level.toLowerCase() === level),
            getYear: (data) => data.filter((item)=> item.publish.indexOf(year) >-1),
        }
        const { getAll, getWord, getLevel, getYear } = rules;
        const filtrateList = [
            keyword === '' ? getAll : getWord,
            level === 'all' ? getAll : getLevel,
            year === 'all' ? getAll : getYear,
        ];
        return compose([...filtrateList])(source);
    }

    function filterEvent($, data){
        let keyword = '';
        let level = 'all';
        let year = 'all';

        $(document).on('blur', '#safetyBulletinSearch', function(e) {
            e.preventDefault();
            keyword = $(e.target).val().trim();
            const result = filterData(data, keyword, level, year);
            renderSafetyTable($, result);
        });

        $(document).on('click', '[name^="safetyBtn"]', function(e){
            level = $(e.target).val();
            const result = filterData(data, keyword, level, year);
            renderSafetyTable($, result);
        });

        $(document).on('change', '#yearSelect', function(e){
            year = $(e.target).val();
            const result = filterData(data, keyword, level, year);
            renderSafetyTable($, result);
        });
    }
    // control event
    renderSafetyTable($, $.safetyData);
    filterEvent($, $.safetyData);
}(jQuery));
