<h1>Artificial Neural Networks</h1>
<h3>To Achieve the following tasks:</h3>
<ul>
	<li>Whether the person is wearing sunglasses or not.</li>
	<li>Predict the direction in which the person is looking.</li>
	<li>To identify the person</li>
</ul>

<b>Contributors : </b>
<ul>
<li><b>G V Sandeep</b></li>
<li><b>Snehal Wadhwani</b></li>
</ul>

<h3>Input And Output Encoding</h3>
<small> One hidden layer for all three modes</small>
<table>
	<thead>
		<tr>
			<th></th>
			<th>Input</th>
			<th>Hidden Units</th>
			<th>Output Units</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td><b>Sunglasses Recognizer</b></td>
			<td>960</td>
			<td>4</td>
			<td>1</td>
		</tr>
		<tr>
			<td><b>Face Recognizer</b></td>
			<td>960</td>
			<td>20</td>
			<td>20</td>
		</tr>
		<tr>
			<td><b>Pose Recognizer</b></td>
			<td>960</td>
			<td>6</td>
			<td>4</td>
		</tr>
	</tbody>
</table>

<h3>Accuracy Measures </h3>
<small>Running time includes learning time and testing on both the test files</small>
<table>
	<thead>
		<tr>
			<th></th>
			<th>Test 1</th>
			<th>Test 2</th>
			<th>Running Time</th>
		</tr>
	</thead>
	<tbody>
		<tr>
			<td><b>Sunglasses Recognizer</b></td>
			<td>100.0%</td>
			<td>97.69%</td>
			<td>10.59 s</td>
		</tr>
		<tr>
			<td><b>Face Recognizer</b></td>
			<td>94.44%</td>
			<td>90.00%</td>
			<td>14.1338 s</td>
		</tr>
		<tr>
			<td><b>Pose Recognizer</b></td>
			<td>90.79%</td>
			<td>93.08%</td>
			<td>42.34 s</td>
		</tr>
	</tbody>
</table>