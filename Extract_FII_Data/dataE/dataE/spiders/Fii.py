import scrapy
import datetime


class FiiSpider(scrapy.Spider):
    name = 'Fii'
    allowed_domains = ['fiis.com.br']
    start_urls = ['https://fiis.com.br/lista-de-fundos-imobiliarios/']

    def parse(self, response):
        funds_raw = response.css('.link-tickers-container')
        for funds in funds_raw:
            yield {
                'title': funds.css('.tickerBox__title ::text').get(),
                'tipo': funds.css('.tickerBox__type ::text').get(),
                'descript': funds.css('.tickerBox__desc ::text').get(),
                'date year': datetime.date.today().year,
                'date month': datetime.date.today().month,
                'date day': datetime.date.today().day
            }
