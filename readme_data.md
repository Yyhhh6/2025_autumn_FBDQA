# 数据说明


-----------

1. 文件命名：
	所有行情文件以`snapshot_sym<xx>_date<yy>_am/pm.csv`命名，代表第xx只证券标的在第yy天的上午/下午的行情数据。
	
2. 数据字段

	| 字段         | 含义               | 说明                                                         |
	| ------------ | ------------------ | ------------------------------------------------------------ |
	| date         | 日期               | sequantial标号：既保留跨标的的可比性，也隐去实际时间         |
	| time         | 时间戳             | 保留实际时间戳，3s一档行情                                   |
	| sym          | 标的(仅序号)       |                                                              |
	| close        | 最新价/收盘价      | 以涨跌幅表示                                                 |
	| amount_delta | 成交量变化         | 从上个tick到当前tick发生的成交金额                           |
	| n_midprice   | 中间价             | 标准化后的中间价，以涨跌幅表示                               |
	| n_bid1       | 买一价             | 标准化后的买一价，以下类似                                   |
	| n_bsize1     | 买一量             |                                                              |
	| n_bid2       | 买二价             |                                                              |
	| n_bsize2     | 买二量             |                                                              |
	| n_bid3       | 买三价             |                                                              |
	| n_bsize3     | 买三量             |                                                              |
	| n_bid4       | 买四价             |                                                              |
	| n_bsize4     | 买四量             |                                                              |
	| n_bid5       | 买五价             |                                                              |
	| n_bsize5     | 买五量             |                                                              |
	| n_ask1       | 卖一价             |                                                              |
	| n_asize1     | 卖一量             |                                                              |
	| n_ask2       | 卖二价             |                                                              |
	| n_asize2     | 卖二量             |                                                              |
	| n_ask3       | 卖三价             |                                                              |
	| n_asize3     | 卖三量             |                                                              |
	| n_ask4       | 卖四价             |                                                              |
	| n_asize4     | 卖四量             |                                                              |
	| n_ask5       | 卖五价             |                                                              |
	| n_asize5     | 卖五量             |                                                              |
	| label5       | 5tick价格移动方向  | 当前tick中间价相对于5tick之前的移动方向，0为下跌，1为不变，2为上涨 |
	| label10      | 10tick价格移动方向 |                                                              |
	| label20      | 20tick价格移动方向 |                                                              |
	| label40      | 40tick价格移动方向 |                                                              |
	| label60      | 60tick价格移动方向 |                                                              |
	
3. 数据处理说明：
	- 中间价说明：
		- 当买一卖一均不为0时，$n\_midprice = \frac{(n\_bid1 + n\_ask1)}{2}$
		- 当有一方为0时，中间价取不为0的价格
		- 否则置为NA
	- 价格移动方向标注说明：
		- 和通常定义不同，为了计算方便，这里以涨跌幅为基准
		- 认定方法：
			- 对每一行，若之后$N$个tick的价格较当前tick价格的涨跌幅上升超过$\alpha$，则认为上涨，标注为2；
			- 若下降幅度超过$\alpha$，则认为下跌，标注为0；
			- 否则认为价格不变，标注为1
		- 计算：
			- $Label_t^N = \phi (n\_midprice_{t+N}-n\_midprice_{t})$ 
			- 其中

				$$
				\begin{equation} 
				\phi(x)=\left \{ 
				\begin{array}{rcl}
				0 & & {x < -\alpha}\\
				1 & & {-\alpha \leq 0 \leq \alpha}\\
				2 & & {\alpha < x}\\
				\end{array}
				\right. 
				\end{equation}
				$$

			-   当N=5，10时，$\alpha = 0.0005$
			-   当N=20，40，60时，$\alpha = 0.001$
